"""Audit the frozen five-seed M3 experiment; summarize Dev without opening Test.

Across-seed SD is sample SD (ddof=1). Day-level paired distributions are kept
separate from training-seed variation; selected-Dev results are not held-out Test.
"""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import torch

from medical_uav.evaluation.evaluate import LOADS
from medical_uav.splits import DEV_SEEDS
from medical_uav.training.io import write_json
from medical_uav.training.trainer import load_checkpoint

STEPS = tuple(range(0, 1_024_001, 102_400))
SEEDS = (1, 2, 3, 4, 5)
COLORS = ("#225E9A", "#A97916", "#C45B2C", "#687D3A", "#A34C77")
EXPECTED_DAYS = {(load, day) for load in LOADS for day in DEV_SEEDS}
DIAGNOSTICS = ("policy_loss", "value_loss", "entropy", "approx_kl", "final_approx_kl",
               "gradient_norm", "advantage_mean", "advantage_std", "return_mean", "return_std")


def read(path):
    return json.loads(Path(path).read_text())


def records(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_tensors(value):
    if isinstance(value, dict):
        for child in value.values():
            finite_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            finite_tensors(child)
    elif isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all(), "non-finite checkpoint tensor"


def day_index(rows):
    index = {(r["episode_load"], r["demand_seed"]): r for r in rows}
    assert len(rows) == len(index) == 120 and set(index) == EXPECTED_DAYS
    for row in rows:
        assert row["split"] == "dev" and row["map_seed"] == 0
        assert row["complete_day"] and row["invalid_actions"] == 0
        assert math.isclose(row["on_time_rate"], row["on_time"] / row["released"]
                            if row["released"] else 0.0, abs_tol=1e-12)
        assert row["released"] == row["delivered"] + row["expired"]
        assert sum(t["released"] for t in row["type_counts"].values()) == row["released"]
    return index


def summarize_days(rows):
    by_load = {}
    for load in LOADS:
        group = [r for r in rows if r["episode_load"] == load]
        by_load[load] = {key: mean(r[key] for r in group) for key in (
            "on_time_rate", "served_rate", "expiry_rate", "reward", "km_flown", "charge_minutes", "wait_minutes")}
    return {"by_load": by_load, "mean_on_time_rate": mean(r["on_time_rate"] for r in by_load.values()),
            "mean_reward": mean(r["reward"] for r in by_load.values())}


def seed_stats(rows):
    answer = {}
    for load in (*LOADS, "mean"):
        values = [r["mean_on_time_rate"] if load == "mean" else r["by_load"][load]["on_time_rate"] for r in rows]
        answer[load] = {"mean": mean(values), "sample_sd": stdev(values), "min": min(values), "max": max(values)}
    return answer


def point_stats(curves, step):
    selected = [next(r for r in curves[seed] if r["transitions"] == step) for seed in SEEDS]
    return {"transitions": step, "rates": seed_stats(selected)}


def analyze(root, output):
    manifest = read(root / "manifest.json")
    protocol = manifest["protocol"]
    assert manifest["status"] == "complete" and manifest["completed_seeds"] == list(SEEDS)
    assert protocol["learner_seeds"] == list(SEEDS) and protocol["evaluation_transitions"] == list(STEPS)
    repo = Path(__file__).resolve().parents[1] if Path(__file__).parent.name == "scripts" else Path.cwd()
    for relative, expected in protocol["frozen_file_sha256"].items():
        assert digest(repo / relative) == expected, f"frozen file changed: {relative}"
    reference = repo / protocol["baseline_reference"]["run_directory"]
    baseline_rows = [r for r in records(reference / "evaluation_episodes.jsonl")
                     if r["transitions"] == 0 and r["policy"] != "ppo"]
    baseline_indices = {policy: day_index([r for r in baseline_rows if r["policy"] == policy])
                        for policy in ("random", "edf", "slack")}
    baselines = {policy: summarize_days(list(index.values())) for policy, index in baseline_indices.items()}
    curves, best_rows, last_rows, run_audits, paired = {}, [], [], {}, []
    for seed in SEEDS:
        run = root / f"seed_{seed}"
        metadata, config, summary = (read(run / name) for name in ("metadata.json", "config.json", "summary.json"))
        assert metadata["git_commit"] == manifest["source_commit"] and not metadata["git_dirty"]
        assert metadata["resume_mode"] == "fresh" and metadata["resume_from"] is None
        assert metadata["learning_rate_schedule"] == "constant"
        assert config == read(repo / f"configs/ppo_m3_seed{seed}.json")
        assert summary["status"] == "complete" and summary["transitions"] == 1_024_000 and summary["updates"] == 1000
        eval_summaries = list(records(run / "evaluations.jsonl"))
        assert [r["transitions"] for r in eval_summaries] == list(STEPS)
        all_days = list(records(run / "evaluation_episodes.jsonl"))
        assert len(all_days) == 1680
        checkpoints = []
        indexed_days = {}
        initial_parameter_digest = None
        for point in eval_summaries:
            step = point["transitions"]
            days = [r for r in all_days if r["policy"] == "ppo" and r["transitions"] == step]
            index = day_index(days)
            checked = summarize_days(days)
            logged = point["policies"]["ppo"]
            assert math.isclose(checked["mean_on_time_rate"], logged["mean_on_time_rate"], abs_tol=1e-12)
            assert math.isclose(checked["mean_reward"], logged["mean_reward"], abs_tol=1e-12)
            checkpoint = load_checkpoint(run / point["checkpoint"])
            assert checkpoint["steps"] == step and checkpoint["updates"] == step // 1024
            assert checkpoint["source_commit"] == manifest["source_commit"]
            finite_tensors(checkpoint["model_state_dict"])
            finite_tensors(checkpoint["optimizer_state_dict"])
            if step == 0:
                initial_parameter_digest = hashlib.sha256(b"".join(
                    value.detach().contiguous().numpy().tobytes()
                    for _, value in sorted(checkpoint["model_state_dict"].items()))).hexdigest()
            checkpoints.append({"transitions": step, "checkpoint": point["checkpoint"],
                                "checkpoint_sha256": digest(run / point["checkpoint"]), **checked})
            indexed_days[step] = index
        # EDF/Slack are deterministic: independently verify the formal-run repeats.
        for policy in ("edf", "slack"):
            actual = day_index([r for r in all_days if r["policy"] == policy])
            assert all(actual[key]["on_time_rate"] == baseline_indices[policy][key]["on_time_rate"]
                       and actual[key]["reward"] == baseline_indices[policy][key]["reward"] for key in EXPECTED_DAYS)
        selected = max(checkpoints, key=lambda r: (r["mean_on_time_rate"], r["mean_reward"]))
        model_best, model_last = load_checkpoint(run / "best.pt"), load_checkpoint(run / "last.pt")
        assert model_best["steps"] == selected["transitions"] and model_last["steps"] == 1_024_000
        candidate_best = load_checkpoint(run / selected["checkpoint"])
        candidate_last = load_checkpoint(run / checkpoints[-1]["checkpoint"])
        assert all(torch.equal(v, candidate_best["model_state_dict"][k]) for k, v in model_best["model_state_dict"].items())
        assert all(torch.equal(v, candidate_last["model_state_dict"][k]) for k, v in model_last["model_state_dict"].items())
        curves[seed] = checkpoints
        best_rows.append({"learner_seed": seed, **selected})
        last_rows.append({"learner_seed": seed, **checkpoints[-1]})
        for selection, point in (("best", selected), ("last", checkpoints[-1])):
            for (load, day), row in sorted(indexed_days[point["transitions"]].items()):
                slack = baseline_indices["slack"][(load, day)]
                paired.append({"selection": selection, "learner_seed": seed, "transitions": point["transitions"],
                               "load": load, "demand_seed": day, "ppo_on_time_rate": row["on_time_rate"],
                               "slack_on_time_rate": slack["on_time_rate"],
                               "delta": row["on_time_rate"] - slack["on_time_rate"]})
        updates = list(records(run / "updates.jsonl"))
        assert len(updates) == 1000 and [r["transitions"] for r in updates] == list(range(1024, 1_024_001, 1024))
        assert all(r["invalid_actions"] == 0 and r["learning_rate"] == 3e-4 for r in updates)
        assert all(isinstance(r[k], (float, int)) and math.isfinite(r[k]) for r in updates for k in DIAGNOSTICS)
        assert max(r["initial_ratio_max_error"] for r in updates) < 1e-5
        completed, completed_decisions, training_sum = Counter(), Counter(), defaultdict(Counter)
        for row in records(run / "episodes.jsonl"):
            assert row["complete_day"] and row["invalid_actions"] == 0 and 0 <= row["demand_seed"] < 100_000
            load = row["episode_load"]
            completed[load] += 1
            completed_decisions[load] += row["decisions"]
            training_sum[load]["reward"] += row["reward"]
            training_sum[load]["on_time_rate"] += row["on_time_rate"]
        assert dict(completed) == summary["episodes_by_load"]
        assert sum(summary["started_episodes_by_load"].values()) == sum(completed.values()) + 8
        partial = model_last["collector"]["partial_days_discarded_on_resume"]
        for load in LOADS:
            assert summary["transitions_by_load"][load] == completed_decisions[load] + sum(
                r["decisions"] for r in partial if r["episode_load"] == load)
        assert sum(summary["transitions_by_load"].values()) == 1_024_000
        run_audits[str(seed)] = {
            "transitions": 1_024_000, "updates": 1000, "ppo_dev_episodes": 1320,
            "source_commit": metadata["git_commit"], "source_manifest_sha256": digest(run / "source_manifest.json"),
            "source_clean": True, "fresh_initialization": True, "test_episodes": 0,
            "initial_model_parameter_sha256": initial_parameter_digest,
            "diagnostics": {key: {"mean": mean(r[key] for r in updates), "min": min(r[key] for r in updates),
                                  "max": max(r[key] for r in updates)} for key in DIAGNOSTICS},
            "initial_ratio_max_error": max(r["initial_ratio_max_error"] for r in updates),
            "kl_early_stop_updates": sum(r["kl_early_stop"] for r in updates),
            "optimizer_steps": sum(r["optimizer_steps"] for r in updates),
            "invalid_actions": 0, "completed_training_days": sum(completed.values()),
            "training_mix": {load: {"opened_days": summary["started_episodes_by_load"][load],
                                    "completed_days": completed[load],
                                    "transitions": summary["transitions_by_load"][load],
                                    "mean_completed_day_reward": training_sum[load]["reward"] / completed[load],
                                    "mean_completed_day_on_time_rate": training_sum[load]["on_time_rate"] / completed[load]}
                             for load in LOADS},
            "timings": {key: summary[key] for key in ("training_seconds", "collect_seconds", "gae_seconds", "update_seconds",
                                                       "eval_seconds", "checkpoint_seconds", "other_seconds", "wall_seconds")}}
        print(f"Audited seed {seed}: best {selected['transitions']:,}, score {selected['mean_on_time_rate']:.6f}", flush=True)
    assert len({r["initial_model_parameter_sha256"] for r in run_audits.values()}) == 5
    mean_curve = [point_stats(curves, step) for step in STEPS]
    paired_summary = {}
    for selection in ("best", "last"):
        by_load = {}
        for load in LOADS:
            selected_pairs = [r for r in paired if r["selection"] == selection and r["load"] == load]
            assert len(selected_pairs) == 200
            per_seed = {}
            for seed in SEEDS:
                deltas = [r["delta"] for r in selected_pairs if r["learner_seed"] == seed]
                per_seed[str(seed)] = {"mean_delta": mean(deltas), "win_days": sum(x > 1e-12 for x in deltas),
                                       "tie_days": sum(abs(x) <= 1e-12 for x in deltas), "loss_days": sum(x < -1e-12 for x in deltas)}
            delta_by_day = [[r["delta"] for r in selected_pairs if r["demand_seed"] == day] for day in DEV_SEEDS]
            values = [r["mean_delta"] for r in per_seed.values()]
            by_load[load] = {"per_seed": per_seed, "mean_delta": mean(values), "sample_sd_across_seeds": stdev(values),
                             "days_all_five_seeds_win": sum(all(v > 1e-12 for v in ds) for ds in delta_by_day),
                             "days_all_five_seeds_lose": sum(all(v < -1e-12 for v in ds) for ds in delta_by_day),
                             "days_any_seed_wins": sum(any(v > 1e-12 for v in ds) for ds in delta_by_day)}
        paired_summary[selection] = by_load
    tail = {}
    for seed in SEEDS:
        curve = curves[seed]
        tail[str(seed)] = {"gain_102400_to_last_pp": 100 * (curve[-1]["mean_on_time_rate"] - curve[1]["mean_on_time_rate"]),
                           "last_interval_gain_pp": 100 * (curve[-1]["mean_on_time_rate"] - curve[-2]["mean_on_time_rate"]),
                           "last_two_intervals_gain_pp": 100 * (curve[-1]["mean_on_time_rate"] - curve[-3]["mean_on_time_rate"]),
                           "best_minus_last_pp": 100 * (best_rows[seed - 1]["mean_on_time_rate"] - curve[-1]["mean_on_time_rate"]),
                           "best_to_last_reward_change": curve[-1]["mean_reward"] - best_rows[seed - 1]["mean_reward"],
                           "best_to_last_by_load": {load: {
                               key: curve[-1]["by_load"][load][key] - best_rows[seed - 1]["by_load"][load][key]
                               for key in ("on_time_rate", "served_rate", "expiry_rate", "reward")}
                               for load in LOADS}}
    result = {
        "milestone": protocol["milestone"], "source_commit": manifest["source_commit"],
        "protocol_sha256": manifest["protocol_sha256"], "run_directory": str(root.resolve()),
        "learner_seeds": list(SEEDS), "best_per_seed": best_rows, "last_per_seed": last_rows,
        "best_across_seed_stats": seed_stats(best_rows), "last_across_seed_stats": seed_stats(last_rows),
        "curves": {str(k): v for k, v in curves.items()}, "pointwise_mean_and_sample_sd": mean_curve,
        "baselines": baselines, "baseline_reference": protocol["baseline_reference"],
        "paired_summary": paired_summary, "tail_changes": tail,
        "seeds_best_before_last": [r["learner_seed"] for r in best_rows if r["transitions"] != 1_024_000],
        "run_audits": run_audits,
        "coverage": {"transitions": 5_120_000, "updates": 5000, "ppo_dev_episodes": 6600,
                     "evaluation_checkpoints": 55, "test_episodes": 0,
                     "complete_training_days": sum(r["completed_training_days"] for r in run_audits.values()),
                     "nonfinite_recorded_training_diagnostics": 0, "invalid_actions": 0},
        "timings": {key: sum(r["timings"][key] for r in run_audits.values()) for key in (
            "training_seconds", "collect_seconds", "gae_seconds", "update_seconds", "eval_seconds",
            "checkpoint_seconds", "other_seconds", "wall_seconds")},
        "orchestrator_wall_seconds": manifest["wall_seconds"],
        "notes": ["All five runs completed the same fixed budget from scratch; seed 0 was excluded.",
                  "Best checkpoints were selected independently using the frozen full-Dev rule; all last models remain available.",
                  "Rates are mean daily rates within load, then equal-weighted across three loads.",
                  "Standard deviation is sample SD across five learner seeds (ddof=1), not across 600 pooled day pairs.",
                  "Random uses the frozen pilot seed-0 canonical baseline; formal-run Random repeats are not cherry-picked.",
                  "Learning curves show all five training repetitions and their pointwise mean.",
                  "Paired day distributions are descriptive Dev comparisons; they do not replace training-seed uncertainty.",
                  "Results are Dev-selected and reused Dev, not held-out Test. No Test cases have been read.",
                  "No causal claim about mixed-load training is possible without the equal-budget Medium-only control."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output.with_suffix(".json"), result)
    write_json(output.parent / (output.name + "_paired.json"), {"unit": "learner_seed x selection x load x Dev day", "pairs": paired})
    draw_curves(result, output.parent / (output.name + "_curves"))
    draw_pairs(paired, output.parent / (output.name + "_paired"))
    write_markdown(result, output.with_suffix(".md"))
    print("Saved audited M3 summary, paired records, curves and result table.")
    return result


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
                         "axes.spines.top": False, "axes.spines.right": False})


def save_figure(fig, stem):
    fig.savefig(stem.with_suffix(".png"), dpi=170, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def draw_curves(result, stem):
    style()
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=.08, right=.97, bottom=.14, top=.78, hspace=.25, wspace=.18)
    x = np.asarray(STEPS) / 1_000_000
    for ax, load, title in zip(axes.flat, (*LOADS, "mean"), ("Low", "Medium", "High", "Equal-load mean")):
        for seed, color in zip(SEEDS, COLORS):
            values = [r["mean_on_time_rate"] if load == "mean" else r["by_load"][load]["on_time_rate"]
                      for r in result["curves"][str(seed)]]
            ax.plot(x, np.asarray(values) * 100, color=color, linewidth=1.1, marker="o", markersize=3,
                    alpha=.8, label=f"Seed {seed}")
        values = [r["rates"][load]["mean"] * 100 for r in result["pointwise_mean_and_sample_sd"]]
        ax.plot(x, values, color="#202830", linewidth=2.6, label="Five-seed mean")
        baseline = result["baselines"]["slack"]
        benchmark = baseline["mean_on_time_rate"] if load == "mean" else baseline["by_load"][load]["on_time_rate"]
        ax.axhline(benchmark * 100, color="#6B7280", linestyle="--", linewidth=1.5, label="Slack-aware")
        ax.set(title=title, ylim=(0, 105), xlim=(-.015, 1.07))
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xticks([0, .2048, .4096, .6144, .8192, 1.024], ["0", ".205", ".410", ".614", ".819", "1.024"])
        ax.grid(axis="y", color="#E4E7EB", linewidth=.7)
        ax.set_axisbelow(True)
        ax.annotate(f"{values[-1]:.1f}%", (x[-1], values[-1]), xytext=(-2, -17),
                    textcoords="offset points", ha="right", fontweight="bold", color="#202830")
    for ax in axes[1]:
        ax.set_xlabel("Training transitions (millions)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean daily on-time rate")
    fig.suptitle("M3: five-seed masked MLP-PPO", x=.08, y=.965, ha="left", fontsize=19, fontweight="bold")
    fig.text(.08, .92, "Scratch seeds 1–5 · 1,024,000 transitions each · unchanged mixed-load protocol", fontsize=11)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.07, .90), ncol=4, frameon=False)
    fig.text(.08, .035, "Dev: 40 fixed days/load, map seed 0. All 11 checkpoints shown for all five learners.\n"
             "The bold line is the pointwise mean, not the mean of independently selected best checkpoints. No Test use.",
             fontsize=10, color="#60666D")
    save_figure(fig, stem)


def draw_pairs(paired, stem):
    style()
    selected = [r for r in paired if r["selection"] == "best"]
    bound = max(abs(r["delta"]) * 100 for r in selected)
    extent = max(10, math.ceil(bound / 10) * 10 + 2)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.8), sharey=True)
    fig.subplots_adjust(left=.08, right=.97, bottom=.22, top=.76, wspace=.15)
    for ax, load in zip(axes, LOADS):
        values = [np.array([r["delta"] * 100 for r in selected if r["load"] == load and r["learner_seed"] == seed]) for seed in SEEDS]
        boxes = ax.boxplot(values, patch_artist=True, showfliers=False, widths=.5,
                           medianprops={"color": "#202830", "linewidth": 1.4})
        for index, (vals, color, box) in enumerate(zip(values, COLORS, boxes["boxes"]), start=1):
            box.set(facecolor=color, alpha=.18, edgecolor=color)
            ax.scatter(index + np.linspace(-.16, .16, len(vals)), vals, color=color, s=11, alpha=.55, linewidth=0)
            ax.scatter(index, mean(vals), color="#202830", marker="D", s=25, zorder=5)
        ax.axhline(0, color="#60666D", linestyle="--", linewidth=1.2)
        ax.set(title=load.title(), ylim=(-extent, extent), xlabel="Learner seed")
        ax.set_xticks(list(SEEDS), [str(s) for s in SEEDS])
        ax.grid(axis="y", color="#E4E7EB", linewidth=.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("PPO minus Slack daily on-time rate (pp)")
    fig.suptitle("Selected-model paired Dev differences", x=.08, y=.955, ha="left", fontsize=18, fontweight="bold")
    fig.text(.08, .88, "Positive values favor PPO · each seed uses its own Dev-selected best checkpoint", fontsize=11)
    fig.text(.08, .07, "40 days/seed/load. Points: days; boxes: quartiles/median; whiskers: 1.5 × IQR; diamonds: mean.\n"
             "Day-level spread is distinct from training-seed uncertainty. These are reused Dev comparisons, not Test.", fontsize=10, color="#60666D")
    save_figure(fig, stem)


def write_markdown(result, path):
    best_stats = result["best_across_seed_stats"]
    final_stats = result["last_across_seed_stats"]
    slack = result["baselines"]["slack"]["mean_on_time_rate"]
    gap = 100 * (best_stats["mean"]["mean"] - slack)
    lines = ["# M3 — Multi-seed MLP-PPO Baseline", "", "Five scratch learner seeds; Dev only; full fixed budget for every run.", "",
             f"Dev-selected models average **{100 * best_stats['mean']['mean']:.2f}% ± {100 * best_stats['mean']['sample_sd']:.2f} pp** across learner seeds, "
             f"versus Slack-aware at {100 * slack:.2f}% (paired mean gap {gap:+.2f} pp). "
             f"Last models average {100 * final_stats['mean']['mean']:.2f}% ± {100 * final_stats['mean']['sample_sd']:.2f} pp.", "",
             "Each cell is mean daily on-time rate (%); aggregate SD is sample SD across five learner seeds.", "",
             "| Seed | Best step | Low | Medium | High | Equal mean | Last equal mean |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for best, last in zip(result["best_per_seed"], result["last_per_seed"]):
        vals = [best["by_load"][load]["on_time_rate"] * 100 for load in LOADS]
        lines.append(f"| {best['learner_seed']} | {best['transitions']:,} | " + " | ".join(f"{v:.2f}" for v in vals)
                     + f" | {100 * best['mean_on_time_rate']:.2f} | {100 * last['mean_on_time_rate']:.2f} |")
    stats = result["best_across_seed_stats"]
    lines.append("| Mean ± SD | — | " + " | ".join(f"{100 * stats[k]['mean']:.2f} ± {100 * stats[k]['sample_sd']:.2f}" for k in (*LOADS, "mean")) + " | — |")
    for policy, name in (("slack", "Slack-aware"), ("edf", "EDF"), ("random", "Random, fixed seed-0 baseline")):
        baseline = result["baselines"][policy]
        lines.append(f"| {name} | — | " + " | ".join(f"{100 * baseline['by_load'][k]['on_time_rate']:.2f}" for k in LOADS)
                     + f" | {100 * baseline['mean_on_time_rate']:.2f} | — |")
    lines += ["", "## Last checkpoints at 1,024,000 transitions", "",
              "| Seed | Low | Medium | High | Equal mean | Best minus last (pp) |",
              "|---:|---:|---:|---:|---:|---:|"]
    for row in result["last_per_seed"]:
        seed = row["learner_seed"]
        lines.append(f"| {seed} | " + " | ".join(f"{100 * row['by_load'][load]['on_time_rate']:.2f}" for load in LOADS)
                     + f" | {100 * row['mean_on_time_rate']:.2f} | {result['tail_changes'][str(seed)]['best_minus_last_pp']:.2f} |")
    lines.append("| Mean ± SD | " + " | ".join(f"{100 * final_stats[k]['mean']:.2f} ± {100 * final_stats[k]['sample_sd']:.2f}" for k in (*LOADS, "mean")) + " | — |")
    lines += ["", "## Learning process", "",
              "![Five seed learning curves and pointwise mean](m3_mlp_ppo_curves.png)", "",
              "The bold curve is the pointwise mean at a common training budget, not an average of Dev-selected checkpoints.", "",
              "| Load | Mean at 102,400 | Mean at 1,024,000 | Increase (pp) | Final interval change (pp) |",
              "|---|---:|---:|---:|---:|"]
    for load in (*LOADS, "mean"):
        start = result["pointwise_mean_and_sample_sd"][1]["rates"][load]["mean"]
        end = result["pointwise_mean_and_sample_sd"][-1]["rates"][load]["mean"]
        previous = result["pointwise_mean_and_sample_sd"][-2]["rates"][load]["mean"]
        lines.append(f"| {load.title()} | {100 * start:.2f}% | {100 * end:.2f}% | {100 * (end-start):+.2f} | {100 * (end-previous):+.2f} |")
    lines += ["", "Best checkpoints before the final budget: " + (", ".join(str(s) for s in result["seeds_best_before_last"]) or "none") + ".", "",
              "The recorded tail changes describe the observed checkpoints. Small final changes or earlier best checkpoints do not prove convergence.", "",
              "## Same-day differences against Slack-aware", "",
              "![Selected model day-level differences](m3_mlp_ppo_paired.png)", "",
              "| Load | Mean difference (pp) | SD across learner-seed mean differences (pp) | Days won by all five | Days lost by all five |",
              "|---|---:|---:|---:|---:|"]
    for load in LOADS:
        p = result["paired_summary"]["best"][load]
        lines.append(f"| {load.title()} | {100 * p['mean_delta']:+.2f} | {100 * p['sample_sd_across_seeds']:.2f} | {p['days_all_five_seeds_win']}/40 | {p['days_all_five_seeds_lose']}/40 |")
    most_degraded = max(result["tail_changes"], key=lambda seed: result["tail_changes"][seed]["best_minus_last_pp"])
    tradeoff = result["tail_changes"][most_degraded]
    lines += ["", f"The largest best-to-last loss is seed {most_degraded}: {tradeoff['best_minus_last_pp']:.2f} pp, "
              f"while equal-load daily reward changed {tradeoff['best_to_last_reward_change']:+.3f}. "
              "Per-load changes in served rate, expiry rate, on-time rate and reward are retained in the JSON. "
              "Reward/on-time divergence is an observed tradeoff to investigate, not proof of a numerical failure or a single behavioral mechanism.", "",
              "Boxes describe day-level variation. The table's SD describes learner-seed variation; these are distinct uncertainty sources.", "",
              "## Sampling, health and elapsed time", "",
              "| Seed | Complete training days | Low / Medium / High transitions | Training (s) | Dev (s) | Wall (s) |",
              "|---:|---:|---|---:|---:|---:|"]
    for seed in SEEDS:
        audit = result["run_audits"][str(seed)]
        counts = " / ".join(f"{audit['training_mix'][load]['transitions']:,}" for load in LOADS)
        time = audit["timings"]
        lines.append(f"| {seed} | {audit['completed_training_days']:,} | {counts} | {time['training_seconds']:.2f} | {time['eval_seconds']:.2f} | {time['wall_seconds']:.2f} |")
    lines += ["", f"All 5,000 updates completed; illegal actions and non-finite recorded diagnostics were zero. "
              f"All 55 archived model/optimizer states passed finite-tensor checks. There were {result['coverage']['complete_training_days']:,} complete training episodes.", "",
              f"Total serial-run wall time: {result['timings']['wall_seconds']:.2f} s; "
              f"training stages {result['timings']['training_seconds']:.2f} s; "
              f"Dev evaluation/logging {result['timings']['eval_seconds']:.2f} s. "
              f"Orchestrator elapsed time: {result['orchestrator_wall_seconds']:.2f} s.", ""]
    lines += ["", "## Artifacts and scope", "", f"Frozen training source commit: `{result['source_commit']}`.", "",
              "All 55 complete Dev checkpoints, best/last models, day-level and update-level records are retained in the raw run directories.",
              "The summary JSON preserves every seed/checkpoint, sampling counts, diagnostics, timings and paired statistics. The paired JSON preserves all 1,200 best/last day pairs.", "",
              "Results are Dev-selected, with no Test use. No inference about the causal effect of mixed loads is supported without an equal-budget Medium-only control.", ""]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/m3_mlp_ppo"))
    args = parser.parse_args()
    analyze(args.run_dir, args.output)
