"""Continuous CPU PPO with bounded logs, fixed Dev selection and safe checkpoints."""

import hashlib
import importlib.metadata
import json
import platform
import random
import shutil
import subprocess
import tarfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from medical_uav.agents.networks import build_network, NETWORKS
from medical_uav.algorithms.ppo import PPOConfig, PPOUpdater
from medical_uav.env import MedicalUAVEnv
from medical_uav.env.config import EnvConfig
from medical_uav.evaluation.evaluate import evaluate, selection_score
from medical_uav.training.collector import Collector
from medical_uav.training.io import append_json, restore_rng, rng_state, write_json


@dataclass
class TrainConfig:
    architecture: str = "mlp"
    learning_rate_schedule: str = "constant"
    total_steps: int = 10_240
    num_envs: int = 8
    rollout_steps: int = 128
    learner_seed: int = 0
    loads: tuple = ("medium",)
    train_seeds: tuple | None = None
    map_seed: int = 0
    threads: int = 1
    eval_split: str = "dev"
    eval_interval_updates: int = 100
    evaluate_initial: bool = True
    evaluate_baselines: bool = True
    ppo: PPOConfig = field(default_factory=PPOConfig)

    def validate(self):
        self.ppo.validate()
        if self.learning_rate_schedule not in ("constant", "late_linear_10pct"):
            raise ValueError("unknown learning rate schedule")
        if self.architecture not in NETWORKS:
            raise ValueError("unknown architecture")
        if min(self.total_steps, self.num_envs, self.rollout_steps, self.threads,
               self.eval_interval_updates) < 1 or self.learner_seed < 0 or self.map_seed < 0:
            raise ValueError("positive budgets/counts and nonnegative seeds required")
        if self.total_steps % (self.num_envs * self.rollout_steps):
            raise ValueError("total_steps must be a multiple of num_envs * rollout_steps")
        if self.eval_split not in ("train", "dev"):
            raise ValueError("Test cannot be used by training or checkpoint selection")
        if not self.loads or set(self.loads) - {"low", "medium", "high"} or len(set(self.loads)) != len(self.loads):
            raise ValueError("loads must be a unique nonempty subset of low/medium/high")
        if self.train_seeds is not None and (not self.train_seeds or any(
                not 0 <= s < 100_000 for s in self.train_seeds)):
            raise ValueError("training days must belong to Train")

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["ppo"] = PPOConfig(**data.get("ppo", {}))
        for field_name in ("loads", "train_seeds"):
            if data.get(field_name) is not None:
                data[field_name] = tuple(data[field_name])
        return cls(**data)


def learning_rate_at(config, update_end_steps):
    """Predeclared schedule indexed by completed transitions at this update end.

    Updates 1..500 retain the original LR at a 1,024,000-step budget. Update
    501 begins linear decay; update 1000 uses 10% of the original LR.
    """
    if config.learning_rate_schedule == "constant":
        return config.ppo.learning_rate
    progress = min(1.0, update_end_steps / config.total_steps)
    factor = 1.0 if progress <= .5 else 1.0 - .9 * (progress - .5) / .5
    return config.ppo.learning_rate * factor


def load_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format_version") != 1 or state.get("architecture") not in NETWORKS:
        raise ValueError("unsupported training checkpoint")
    configured = TrainConfig.from_dict(state["config"]).architecture
    if NETWORKS[configured].architecture != state["architecture"]:
        raise ValueError("checkpoint architecture disagrees with its configuration")
    return state


def _save_checkpoint(path, network, updater, collector, config, steps, updates, best_score):
    state = {
        "format_version": 1, "architecture": network.architecture,
        "model_state_dict": network.state_dict(), "optimizer_state_dict": updater.optimizer.state_dict(),
        "config": asdict(config), "env_config": asdict(EnvConfig(map_seed=config.map_seed)),
        "steps": steps, "updates": updates, "best_score": best_score,
        "rng": rng_state(), "updater_rng": updater.rng.bit_generator.state,
        "collector": collector.state_dict(), "resume_mode": "new_train_days",
        "torch_version": str(torch.__version__),
        "source_commit": json.loads((path.parent / "metadata.json").read_text())["git_commit"],
        "source_manifest_sha256": hashlib.sha256((path.parent / "source_manifest.json").read_bytes()).hexdigest(),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _source_record(output):
    repo = Path(__file__).resolve().parents[2]
    def git(*args):
        result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    files = git("ls-files", "--cached", "--others", "--exclude-standard") or ""
    manifest = {}
    with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
        for relative in sorted(set(files.splitlines())):
            source = repo / relative
            if source.is_file() and source.suffix in {".py", ".yml", ".yaml", ".toml", ".txt", ".md"}:
                manifest[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
                archive.add(source, arcname=relative, recursive=False)
    write_json(output / "source_manifest.json", manifest)
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
            "source_archive": "source.tar.gz", "source_manifest": "source_manifest.json"}


def train(config, output_dir, resume=None):
    """Run full updates; resume restores RNG/optimizer but discards unfinished days.

    Output directories must be fresh, including for resumed runs. Partial episodes
    discarded on resume are listed in metadata and are never logged as full days.
    """
    config.validate()
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("use a fresh output directory to avoid overwriting run evidence")
    checkpoint = load_checkpoint(resume) if resume else None
    if checkpoint:
        old, new = asdict(TrainConfig.from_dict(checkpoint["config"])), asdict(config)
        old.pop("total_steps")
        new.pop("total_steps")
        if old != new:
            raise ValueError("resuming permits only a changed total_steps budget")
        if checkpoint["steps"] >= config.total_steps:
            raise ValueError("resume budget must exceed the checkpoint's completed steps")
    output.mkdir(parents=True, exist_ok=True)
    if checkpoint and checkpoint["best_score"] is not None:
        inherited = Path(resume).parent / "best.pt"
        if inherited.is_file():
            inherited_state = load_checkpoint(inherited)
            if inherited_state["best_score"] != checkpoint["best_score"]:
                raise ValueError("adjacent best checkpoint does not match the resume checkpoint's selection score")
            shutil.copy2(inherited, output / "best.pt")
        else:
            raise ValueError("the selected best.pt must accompany a Dev training checkpoint when resuming")
    torch.set_num_threads(config.threads)
    random.seed(config.learner_seed)
    np.random.seed(config.learner_seed)
    torch.manual_seed(config.learner_seed)
    started = time.perf_counter()
    network = build_network(config.architecture)
    updater = PPOUpdater(network, config.ppo, config.learner_seed)
    steps, updates, best_score = 0, 0, None
    if checkpoint:
        network.load_state_dict(checkpoint["model_state_dict"])
        updater.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        updater.rng.bit_generator.state = checkpoint["updater_rng"]
        steps, updates, best_score = checkpoint["steps"], checkpoint["updates"], checkpoint["best_score"]
        restore_rng(checkpoint["rng"])
    collector = Collector(
        config.num_envs, config.learner_seed, config.loads, config.train_seeds,
        env_factory=lambda: MedicalUAVEnv(config=EnvConfig(map_seed=config.map_seed)),
        state=checkpoint["collector"] if checkpoint else None)
    metadata = {
        **_source_record(output), "config": asdict(config),
        "env_config": asdict(EnvConfig(map_seed=config.map_seed)),
        "python": platform.python_version(), "torch": str(torch.__version__),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "resume_from": str(Path(resume).resolve()) if resume else None,
        "resume_mode": "new_train_days" if resume else "fresh",
        "discarded_partial_days": checkpoint["collector"]["partial_days_discarded_on_resume"] if checkpoint else [],
        "selection_rule": "full Dev: equal-load mean daily on_time_rate, then mean daily reward",
        "value_loss": "MSE without additional 1/2", "discount": "per decision; gamma=1 by default",
        "learning_rate_schedule": config.learning_rate_schedule,
    }
    write_json(output / "metadata.json", metadata)
    write_json(output / "config.json", asdict(config))
    eval_total, checkpoint_total = 0.0, 0.0
    collect_total, gae_total, update_total = 0.0, 0.0, 0.0
    best_in_this_run = False
    last_evaluation = None

    def run_evaluation(include_baselines=False):
        nonlocal best_score, eval_total, checkpoint_total, best_in_this_run, last_evaluation
        tick = time.perf_counter()
        policies = ("ppo", "random", "edf", "slack") if include_baselines else ("ppo",)
        records, summary = evaluate(network, policies, split=config.eval_split,
                                    learner_seed=config.learner_seed, map_seed=config.map_seed)
        for record in records:
            record.update({"transitions": steps, "update_index": updates})
            append_json(output / "evaluation_episodes.jsonl", record)
        elapsed = time.perf_counter() - tick
        eval_total += elapsed
        checkpoint_name = f"checkpoint_{steps:09d}.pt"
        last_evaluation = {"transitions": steps, "update_index": updates,
                           "checkpoint": checkpoint_name,
                           "split": config.eval_split, "eval_seconds": elapsed, **summary}
        append_json(output / "evaluations.jsonl", last_evaluation)
        if config.eval_split == "dev":
            score = selection_score(summary)
            if best_score is None or tuple(score) > tuple(best_score):
                best_score = score
                tick = time.perf_counter()
                _save_checkpoint(output / "best.pt", network, updater, collector, config, steps, updates, best_score)
                checkpoint_total += time.perf_counter() - tick
                best_in_this_run = True
        tick = time.perf_counter()
        _save_checkpoint(output / checkpoint_name, network, updater, collector, config, steps, updates, best_score)
        checkpoint_total += time.perf_counter() - tick
        return elapsed

    batch_size = config.num_envs * config.rollout_steps
    try:
        if config.evaluate_initial:
            run_evaluation(config.evaluate_baselines)
        while steps < config.total_steps:
            checkpoint_before = checkpoint_total
            tick = time.perf_counter()
            buffer, episodes = collector.collect(network, config.rollout_steps)
            collect_seconds = time.perf_counter() - tick
            collect_total += collect_seconds
            tick = time.perf_counter()
            buffer.finish(config.ppo.gamma, config.ppo.gae_lambda)
            gae_seconds = time.perf_counter() - tick
            gae_total += gae_seconds
            tick = time.perf_counter()
            for group in updater.optimizer.param_groups:
                group["lr"] = learning_rate_at(config, steps + batch_size)
            diagnostics = updater.update(buffer)
            update_seconds = time.perf_counter() - tick
            update_total += update_seconds
            steps += batch_size
            updates += 1
            for record in episodes:
                # Exact aggregate transition position within this synchronous rollout.
                record["transitions"] = steps - batch_size + record["rollout_step"] * config.num_envs + record["env_index"] + 1
                record["update_index"] = updates
                append_json(output / "episodes.jsonl", record)
            eval_seconds = 0.0
            if updates % config.eval_interval_updates == 0 or steps == config.total_steps:
                eval_seconds = run_evaluation()
            tick = time.perf_counter()
            _save_checkpoint(output / "last.pt", network, updater, collector, config, steps, updates, best_score)
            checkpoint_seconds = time.perf_counter() - tick
            checkpoint_total += checkpoint_seconds
            row = {"transitions": steps, "update_index": updates, "learner_seed": config.learner_seed,
                   "invalid_actions": 0, "completed_episodes": len(episodes),
                   "legal_actions_mean": float(buffer.action_mask.sum(-1).float().mean()),
                   "single_legal_fraction": float((buffer.action_mask.sum(-1) == 1).float().mean()),
                   "terminated_count": int(buffer.terminated.sum()), "truncated_count": int(buffer.truncated.sum()),
                   "elapsed_minutes_mean": float(buffer.elapsed.mean()),
                   "collect_seconds": collect_seconds, "gae_seconds": gae_seconds,
                   "update_seconds": update_seconds, "eval_seconds": eval_seconds,
                   "checkpoint_seconds": checkpoint_total - checkpoint_before,
                   "wall_seconds": time.perf_counter() - started,
                   "learning_rate": updater.optimizer.param_groups[0]["lr"],
                   "transitions_by_load": dict(collector.transitions_by_load),
                   "started_episodes_by_load": dict(collector.started_episodes_by_load),
                   "episodes_by_load": dict(collector.episodes_by_load), **diagnostics}
            append_json(output / "updates.jsonl", row)
            print(f"update {updates}: {steps} transitions, value_loss={diagnostics['value_loss']:.3f}, "
                  f"KL={diagnostics['final_approx_kl']:.5f}, episodes={len(episodes)}", flush=True)
    except Exception as error:
        # Failed rollout data is diagnostic only and is never used for an update.
        write_json(output / "failure.json", {"error": repr(error), "last_completed_steps": steps,
                                             "last_completed_update": updates})
        torch.save({"observations": collector.obs, "masks": collector.masks,
                    "collector": collector.state_dict()}, output / "failure_inputs.pt")
        raise
    finally:
        collector.close()
    result = {"status": "complete", "transitions": steps, "updates": updates,
              "learner_seed": config.learner_seed, "wall_seconds": time.perf_counter() - started,
              "eval_seconds": eval_total, "checkpoint_seconds": checkpoint_total,
              "collect_seconds": collect_total, "gae_seconds": gae_total,
              "update_seconds": update_total,
              "training_seconds": collect_total + gae_total + update_total,
              "learning_rate_schedule": config.learning_rate_schedule,
              "best_score": best_score, "best_written_in_this_run": best_in_this_run,
              "inherited_best_checkpoint": str(Path(resume).parent / "best.pt") if resume and not best_in_this_run else None,
              "last_evaluation": last_evaluation, "transitions_by_load": dict(collector.transitions_by_load),
              "episodes_by_load": dict(collector.episodes_by_load), "resume_mode": metadata["resume_mode"]}
    result["started_episodes_by_load"] = dict(collector.started_episodes_by_load)
    result["started_counts_include_history"] = collector.started_counts_include_history
    result["other_seconds"] = result["wall_seconds"] - eval_total - checkpoint_total - result["training_seconds"]
    write_json(output / "summary.json", result)
    return result
