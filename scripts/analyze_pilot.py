"""Audit the five-point mixed pilot and export its scientific learning curve.

Uses the recorded complete Dev days as authority, independently recomputing daily
rates and equal-load means. This script does not train, tune or read Test results.
"""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import torch

from medical_uav.evaluation.evaluate import LOADS
from medical_uav.splits import DEV_SEEDS
from medical_uav.training.io import write_json
from medical_uav.training.trainer import load_checkpoint

STEPS = (0, 25_600, 51_200, 76_800, 102_400)


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_finite_tensors(value):
    if isinstance(value, dict):
        for child in value.values():
            assert_finite_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            assert_finite_tensors(child)
    elif isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all(), "non-finite checkpoint tensor"


def analyze(run, output):
    config, metadata, summary = (read(run / name) for name in ("config.json", "metadata.json", "summary.json"))
    updates, episodes = rows(run / "updates.jsonl"), rows(run / "episodes.jsonl")
    evaluations, days = rows(run / "evaluations.jsonl"), rows(run / "evaluation_episodes.jsonl")
    assert summary["status"] == "complete" and config["total_steps"] == 102_400
    assert config["learner_seed"] == 0 and config["loads"] == list(LOADS)
    assert config["num_envs"] == 8 and config["rollout_steps"] == 128
    assert config["threads"] == 1 and config["eval_interval_updates"] == 25
    assert metadata["resume_mode"] == "fresh" and metadata["resume_from"] is None
    assert not metadata["git_dirty"] and config["map_seed"] == 0
    assert [r["transitions"] for r in evaluations] == list(STEPS)
    assert [r["transitions"] for r in updates] == list(range(1024, 102_401, 1024))
    assert len(updates) == 100
    assert all(r["split"] == "dev" and r["complete_day"] and r["invalid_actions"] == 0 for r in days)
    assert all(r["complete_day"] and r["invalid_actions"] == 0 for r in episodes)
    assert all(r["learning_rate"] == config["ppo"]["learning_rate"] == 3e-4 for r in updates)
    diagnostic_keys = ("policy_loss", "value_loss", "entropy", "approx_kl", "final_approx_kl",
                       "gradient_norm", "advantage_mean", "advantage_std", "return_mean", "return_std")
    assert all(isinstance(r[k], (int, float)) and math.isfinite(r[k]) for r in updates for k in diagnostic_keys)
    expected = {(load, seed) for load in LOADS for seed in DEV_SEEDS}
    curve, checkpoint_records = [], {}
    for evaluation in evaluations:
        step = evaluation["transitions"]
        selected = [r for r in days if r["policy"] == "ppo" and r["transitions"] == step]
        assert len(selected) == len(expected) == 120
        assert {(r["episode_load"], r["demand_seed"]) for r in selected} == expected
        for r in selected:
            rate = r["on_time"] / r["released"] if r["released"] else 0.0
            assert math.isclose(rate, r["on_time_rate"], abs_tol=1e-12)
            assert r["map_seed"] == 0
            assert sum(t["released"] for t in r["type_counts"].values()) == r["released"]
        rates = {load: mean(r["on_time_rate"] for r in selected if r["episode_load"] == load) for load in LOADS}
        rewards = {load: mean(r["reward"] for r in selected if r["episode_load"] == load) for load in LOADS}
        logged = evaluation["policies"]["ppo"]
        assert all(math.isclose(rates[load], logged["by_load"][load]["on_time_rate"], abs_tol=1e-12) for load in LOADS)
        assert math.isclose(mean(rates.values()), logged["mean_on_time_rate"], abs_tol=1e-12)
        path = run / evaluation["checkpoint"]
        checkpoint = load_checkpoint(path)
        assert checkpoint["steps"] == step and checkpoint["source_commit"] == metadata["git_commit"]
        assert_finite_tensors(checkpoint["model_state_dict"])
        assert_finite_tensors(checkpoint["optimizer_state_dict"])
        checkpoint_records[step] = selected
        curve.append({"transitions": step, "checkpoint": path.name, "checkpoint_sha256": sha256(path),
                      "on_time_rate": rates, "mean_on_time_rate": mean(rates.values()),
                      "reward": rewards, "mean_reward": mean(rewards.values()),
                      "evaluation_seconds": evaluation["eval_seconds"]})
    best_point = max(curve, key=lambda r: (r["mean_on_time_rate"], r["mean_reward"]))
    best, last = load_checkpoint(run / "best.pt"), load_checkpoint(run / "last.pt")
    assert best["steps"] == best_point["transitions"] and last["steps"] == 102_400
    best_candidate = load_checkpoint(run / best_point["checkpoint"])
    last_candidate = load_checkpoint(run / curve[-1]["checkpoint"])
    assert all(torch.equal(v, best_candidate["model_state_dict"][k])
               for k, v in best["model_state_dict"].items())
    assert all(torch.equal(v, last_candidate["model_state_dict"][k])
               for k, v in last["model_state_dict"].items())
    completed = Counter(r["episode_load"] for r in episodes)
    assert dict(completed) == summary["episodes_by_load"]
    assert sum(summary["transitions_by_load"].values()) == 102_400
    assert summary["started_counts_include_history"]
    assert sum(summary["started_episodes_by_load"].values()) == len(episodes) + 8
    partial = last["collector"]["partial_days_discarded_on_resume"]
    training_mix = {}
    for load in LOADS:
        group = [r for r in episodes if r["episode_load"] == load]
        transitions = summary["transitions_by_load"][load]
        assert transitions == sum(r["decisions"] for r in group) + sum(r["decisions"] for r in partial if r["episode_load"] == load)
        training_mix[load] = {
            "opened_days": summary["started_episodes_by_load"][load], "completed_days": completed[load],
            "active_days": summary["started_episodes_by_load"][load] - completed[load],
            "opened_day_share": summary["started_episodes_by_load"][load] / sum(summary["started_episodes_by_load"].values()),
            "transitions": transitions, "transition_share": transitions / 102_400,
            "mean_completed_day_reward": mean(r["reward"] for r in group),
            "mean_completed_day_on_time_rate": mean(r["on_time_rate"] for r in group),
        }
    slack = {(r["episode_load"], r["demand_seed"]): r for r in days if r["policy"] == "slack"}
    assert set(slack) == expected and len(slack) == 120
    slack_rates = {load: mean(r["on_time_rate"] for (level, _), r in slack.items() if level == load) for load in LOADS}
    comparisons = {}
    for label, point in (("best", best_point), ("last", curve[-1])):
        by_load = {}
        for load in LOADS:
            deltas = [r["on_time_rate"] - slack[(load, r["demand_seed"])]["on_time_rate"]
                      for r in checkpoint_records[point["transitions"]] if r["episode_load"] == load]
            by_load[load] = {"mean_delta_percentage_points": mean(deltas) * 100,
                             "win_days": sum(d > 1e-12 for d in deltas),
                             "tie_days": sum(abs(d) <= 1e-12 for d in deltas),
                             "loss_days": sum(d < -1e-12 for d in deltas)}
        comparisons[label] = by_load
    diagnostics = {k: {"min": min(r[k] for r in updates), "max": max(r[k] for r in updates),
                        "mean": mean(r[k] for r in updates)} for k in diagnostic_keys}
    diagnostics.update({"initial_ratio_max_error": max(r["initial_ratio_max_error"] for r in updates),
                        "explained_variance": {"min": min(r["explained_variance"] for r in updates),
                                               "max": max(r["explained_variance"] for r in updates),
                                               "last": updates[-1]["explained_variance"]},
                        "kl_early_stop_updates": sum(r["kl_early_stop"] for r in updates),
                        "optimizer_steps": sum(r["optimizer_steps"] for r in updates),
                        "invalid_actions": sum(r["invalid_actions"] for r in updates)})
    timings = {k: summary[k] for k in ("collect_seconds", "gae_seconds", "update_seconds", "training_seconds",
                                      "eval_seconds", "checkpoint_seconds", "other_seconds", "wall_seconds")}
    assert math.isclose(timings["wall_seconds"], sum(timings[k] for k in
                        ("training_seconds", "eval_seconds", "checkpoint_seconds", "other_seconds")), abs_tol=1e-9)
    result = {
        "purpose": "Five-checkpoint mixed-load pilot; one learner seed, same-map full Dev only",
        "source_commit": metadata["git_commit"], "config": config,
        "run_directory": str(run.resolve()), "source_manifest_sha256": sha256(run / "source_manifest.json"),
        "curve": curve, "best": best_point, "last": curve[-1],
        "slack_on_time_rate": slack_rates, "slack_mean_on_time_rate": mean(slack_rates.values()),
        "paired_dev_comparison": comparisons, "training_mix": training_mix,
        "diagnostics": diagnostics, "timings": timings,
        "coverage": {"updates": 100, "complete_training_days": len(episodes), "ppo_dev_days": 600,
                     "baseline_dev_days": 360, "test_days": 0, "learner_seeds": [0]},
        "notes": ["Rates are averages over complete days, then equal-weighted across loads.",
                  "Opened days include initial/reset days that may still be active at the budget boundary.",
                  "Training daily averages pool evolving stochastic policies; use Dev for performance comparisons.",
                  "Initial evaluation timing includes three baselines (360 additional Dev days).",
                  "One learner seed does not quantify training-seed uncertainty; no intervals are claimed.",
                  "Increased budget and mixed loads changed together; no causal claim about mixing is supported."],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output.with_suffix(".json"), result)
    make_figure(result, output)
    print("Transitions       Low      Medium       High       Equal mean")
    for point in curve:
        print(f"{point['transitions']:>11,}  " + "  ".join(f"{100 * point['on_time_rate'][load]:8.2f}%" for load in LOADS)
              + f"  {100 * point['mean_on_time_rate']:9.2f}%")
    print(f"Best step: {best_point['transitions']}; wall: {timings['wall_seconds']:.2f}s")
    return result


def make_figure(result, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.08, right=.97, bottom=.13, top=.83, hspace=.30, wspace=.18)
    blue, neutral = "#225E9A", "#60666D"
    x = [r["transitions"] / 1000 for r in result["curve"]]
    for ax, key, title in zip(axes.flat, (*LOADS, "mean"), ("Low", "Medium", "High", "Equal-load mean")):
        values = [(r["mean_on_time_rate"] if key == "mean" else r["on_time_rate"][key]) * 100
                  for r in result["curve"]]
        benchmark = 100 * (result["slack_mean_on_time_rate"] if key == "mean" else result["slack_on_time_rate"][key])
        ax.plot(x, values, color=blue, marker="o", linewidth=2, markersize=6, label="MLP-PPO")
        ax.axhline(benchmark, color=neutral, linestyle="--", linewidth=1.5, label="Slack-aware")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylim(0, 105)
        ax.set_xlim(-2, 108)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.set_xticks(x, ["0", "25.6", "51.2", "76.8", "102.4"])
        ax.grid(axis="y", color="#E4E7EB", linewidth=.7)
        ax.set_axisbelow(True)
        ax.annotate(f"{values[-1]:.1f}%", (x[-1], values[-1]), textcoords="offset points",
                    xytext=(-5, -17), ha="right", color=blue, fontweight="bold")
    for ax in axes[1]:
        ax.set_xlabel("Training transitions (thousands)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean daily on-time rate")
    fig.suptitle("Masked MLP-PPO: mixed-load pilot", x=.08, y=.965, ha="left", fontsize=19, fontweight="bold")
    fig.text(.08, .92, "Scratch initialization · learner seed 0 · equal-probability load per new training day", fontsize=11)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.07, .90), ncol=2, frameon=False)
    fig.text(.08, .035, "Dev: 40 fixed days per load, map seed 0. Five measured checkpoints; no Test use.\n"
             "One training repeat; lines connect observations and do not show training-seed uncertainty.", fontsize=10, color=neutral)
    fig.savefig(output.with_suffix(".png"), dpi=170, facecolor="white")
    fig.savefig(output.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/ppo_mixed_pilot_seed0"), help="output stem")
    args = parser.parse_args()
    analyze(args.run_dir, args.output)
