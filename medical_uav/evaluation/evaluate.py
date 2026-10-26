"""Deterministic neural evaluation, same-day baselines and fixed model selection."""

from collections import Counter

import numpy as np
import torch

from medical_uav.baselines.heuristics import edf_policy, random_policy, slack_aware_policy
from medical_uav.env import MedicalUAVEnv
from medical_uav.env.actions import decode
from medical_uav.evaluation.diagnostics import EpisodeDiagnostics
from medical_uav.env.config import EnvConfig
from medical_uav.metrics import episode_record
from medical_uav.splits import DEV_SEEDS, DEV_EXTRA_SEEDS, TEST_SEEDS, TRAIN_SEED_RANGE
from medical_uav.training.io import isolated_rng

LOADS = ("low", "medium", "high")


def split_seeds(split):
    if split == "dev":
        return tuple(DEV_SEEDS)
    if split == "dev-extra":
        return tuple(DEV_EXTRA_SEEDS)
    if split == "test":
        return tuple(TEST_SEEDS)
    if split == "train":
        return tuple(range(3))  # cheap diagnostics, not checkpoint selection
    raise ValueError("split must be train, dev, dev-extra or test")


def validate_days(seeds, split):
    valid = (range(*TRAIN_SEED_RANGE) if split == "train" else split_seeds(split))
    if not seeds or len(set(seeds)) != len(seeds) or any(s not in valid for s in seeds):
        raise ValueError(f"evaluation days must be unique and belong to {split}")


def summarize(records):
    by_policy = {}
    for policy in sorted({r["policy"] for r in records}):
        by_load = {}
        for load in LOADS:
            rows = [r for r in records if r["policy"] == policy and r["episode_load"] == load]
            if rows:
                by_load[load] = {"episodes": len(rows), **{
                    name: float(np.mean([r[name] for r in rows])) for name in (
                        "on_time_rate", "served_rate", "expiry_rate", "reward",
                        "km_flown", "charge_minutes", "wait_minutes")}}
        by_policy[policy] = {"by_load": by_load,
                             "mean_on_time_rate": float(np.mean([r["on_time_rate"] for r in by_load.values()])),
                             "mean_reward": float(np.mean([r["reward"] for r in by_load.values()]))}
    return {"policies": by_policy, "episodes": len(records),
            "split": records[0]["split"] if records else None,
            "demand_seeds": sorted({r["demand_seed"] for r in records})}


@torch.no_grad()
def evaluate(network=None, policies=("ppo",), seeds=None, split="dev", loads=LOADS,
             learner_seed=0, map_seed=0, env_config=None, diagnostics=False):
    """Evaluate complete days without changing the training random-number stream.

    Each Random baseline day has its own Generator, independent of loop ordering.
    Full Dev is required by the trainer for best-checkpoint selection.
    """
    seeds = tuple(split_seeds(split) if seeds is None else seeds)
    validate_days(seeds, split)
    if not policies or len(set(policies)) != len(policies) or set(policies) - {"ppo", "random", "edf", "slack"}:
        raise ValueError("unknown or empty policy list")
    if not loads or set(loads) - set(LOADS) or len(set(loads)) != len(loads):
        raise ValueError("unknown, duplicate or empty load list")
    if "ppo" in policies and network is None:
        raise ValueError("a network is required for PPO evaluation")
    config = env_config or EnvConfig(map_seed=map_seed)
    records = []
    was_training = network.training if network is not None else None
    with isolated_rng():
        env = MedicalUAVEnv(config=config)
        if network is not None:
            network.eval()
        try:
            for policy in policies:
                for load in loads:
                    for day in seeds:
                        obs, info = env.reset(options={"demand_seed": day, "load": load})
                        rng = np.random.default_rng(np.random.SeedSequence([learner_seed, day, LOADS.index(load), config.map_seed]))
                        observer = EpisodeDiagnostics() if diagnostics else None
                        counts, minutes, decisions = Counter(), Counter(), 0
                        while True:
                            if policy == "ppo":
                                action, _, _ = network.act(torch.from_numpy(obs),
                                                           torch.from_numpy(info["action_mask"]), True)
                                action = int(action)
                            else:
                                action = (random_policy(env, rng) if policy == "random" else
                                          edf_policy(env) if policy == "edf" else slack_aware_policy(env))
                            if not info["action_mask"][action]:
                                raise RuntimeError("illegal evaluation action")
                            if observer is not None:
                                observer.before(env, action)
                            obs, reward, terminated, truncated, info = env.step(action)
                            if observer is not None:
                                observer.after(env, action)
                            if info["invalid_action"] or not np.isfinite(reward):
                                raise RuntimeError("invalid evaluation transition")
                            env.sim.check_invariant()
                            if decisions > 10_000:
                                raise RuntimeError("evaluation day exceeded safety decision cap")
                            kind = decode(action)[0]
                            counts[kind] += 1
                            minutes[kind] += float(info["elapsed"])
                            decisions += 1
                            if terminated or truncated:
                                record = episode_record(env, load, decisions, counts, minutes,
                                                        learner_seed, terminated=terminated, truncated=truncated)
                                if not record["complete_day"]:
                                    raise RuntimeError("evaluation requires completed days")
                                record.update({"policy": policy, "split": split})
                                if observer is not None:
                                    record["diagnostics"] = observer.finish(env)
                                records.append(record)
                                break
        finally:
            env.close()
            if network is not None:
                network.train(was_training)
    return records, summarize(records)


def selection_score(summary):
    """Equal-load mean DAILY on-time rate, with mean reward as the tie-breaker."""
    result = summary["policies"]["ppo"]
    if (set(result["by_load"]) != set(LOADS) or summary.get("split") != "dev"
            or summary.get("demand_seeds") != list(DEV_SEEDS)
            or any(r["episodes"] != len(DEV_SEEDS) for r in result["by_load"].values())):
        raise ValueError("checkpoint selection requires full Dev: 40 days at all three loads")
    return result["mean_on_time_rate"], result["mean_reward"]
