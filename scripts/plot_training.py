"""Learning curves of groups of PPO runs, and how fast each run reaches a fixed score.

Each run's evals.csv holds the greedy on-time rate on the first 20 selection days at medium and
high load (multi-map runs: the days rotate over the selection maps); the score is their mean. The
figure shows each group's mean over its runs with the range as a band, and Slack-aware on the
same days and maps (all runs must share map 0 or the selection maps). The table gives each run's final score, its
mean score over the logged curve, and the decisions at which it first reaches (and keeps at the
next evaluation) 95 % of the final score of the first group's run with the same training seed.
For every other group, the decisions saved against the first group, seed by seed, after adding
the group's extra cost (e.g. demonstrations and value warm-up); a run that never confirms the
target is left out of the savings and counted as censored.

Usage:
    python scripts/plot_training.py "PPO:runs/m3_set_s11,runs/m3_set_s22" \\
        "BC then PPO:runs/m3_set_bc_s11,runs/m3_set_bc_s22" --extra-cost "BC then PPO=178856" \\
        --out outputs/training_curves.png
"""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from medical_uav.baselines.heuristics import slack_aware_policy
from medical_uav.env import EnvConfig
from medical_uav.evaluation import curve_mean, first_sustained_crossing, heuristic, run_day
from medical_uav.splits import SELECT_MAP_SEEDS, SELECT_SEEDS

LOADS = ("medium", "high")
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
TARGET = 0.95  # fraction of the reference run's final score


def read_run(run: str) -> dict:
    """Decisions, per-load on-time rates and their mean from a run's evals.csv; the training
    seed from the run's name (runs/<name>_s<seed>)."""
    rows = list(csv.DictReader(open(Path(run) / "evals.csv")))
    out = {"decisions": np.array([int(r["decisions"]) for r in rows]),
           "seed": int(re.search(r"_s(\d+)$", run.rstrip("/")).group(1))}
    for load in LOADS:
        out[load] = np.array([float(r[f"on_time_{load}"]) for r in rows])
    out["score"] = (out["medium"] + out["high"]) / 2
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("groups", nargs="+", help='"label:run_dir,run_dir,..."; the first group '
                                                  "is the reference for the crossing")
    parser.add_argument("--extra-cost", action="append", default=[],
                        help='"label=decisions" spent by a group outside its PPO training')
    parser.add_argument("--eval-days", type=int, default=20)
    parser.add_argument("--out", default="outputs/training_curves.png")
    args = parser.parse_args()

    groups = {}
    for g in args.groups:
        label, runs = g.split(":")
        groups[label] = {run: read_run(run) for run in runs.split(",")}
    runs = [run for g in groups.values() for run in g]
    many = {json.loads((Path(run) / "run.json").read_text()).get("maps") == "train" for run in runs}
    if len(many) > 1:
        raise SystemExit("single-map and multi-map runs were evaluated on different maps")
    days = list(SELECT_SEEDS)[:args.eval_days]
    maps = [SELECT_MAP_SEEDS[k % len(SELECT_MAP_SEEDS)] if many.pop() else 0
            for k in range(len(days))]  # as train_ppo.py's evaluation
    slack = {load: np.mean([run_day(heuristic(slack_aware_policy), load, s,
                                    EnvConfig(map_seed=m))["on_time_rate"]
                            for s, m in zip(days, maps)]) for load in LOADS}

    fig = Figure(figsize=(9, 3.4), dpi=150)
    for k, load in enumerate(LOADS):
        ax = fig.add_subplot(1, len(LOADS), k + 1)
        for color, (label, runs) in zip(COLORS, groups.items()):
            x = next(iter(runs.values()))["decisions"] / 1e6
            y = np.array([r[load] for r in runs.values()])
            ax.fill_between(x, y.min(0), y.max(0), color=color, alpha=0.15, lw=0)
            ax.plot(x, y.mean(0), color=color, lw=2, label=f"{label} ({len(runs)} seeds)")
        ax.axhline(slack[load], color="#7a7974", ls="--", lw=1.2, label="Slack-aware")
        ax.set_title(f"{load} load")
        ax.set_xlabel("training decisions (millions)")
        ax.set_ylabel(f"on-time rate, {args.eval_days} selection days")
        ax.grid(alpha=0.3, lw=0.5)
        if k == 0:
            ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)

    extra = {label: float(cost) for label, cost in (e.split("=") for e in args.extra_cost)}
    ref_label = next(iter(groups))
    reference = {r["seed"]: r["score"][-1] for r in groups[ref_label].values()}
    print("score = mean of medium and high on-time; "
          f"Slack-aware {np.mean(list(slack.values())):.3f}")
    print(f"{'run':28s} {'final':>6s} {'curve mean':>10s} {'reaches target at':>18s}")
    hits = {}  # group -> seed -> decisions at the confirmed crossing, or None
    for label, runs in groups.items():
        for run, r in runs.items():
            target = TARGET * reference[r["seed"]]
            hit = first_sustained_crossing(r["decisions"], r["score"], target)
            hits.setdefault(label, {})[r["seed"]] = hit
            mean = curve_mean(r["decisions"], r["score"], r["decisions"][0], r["decisions"][-1])
            print(f"{Path(run).name:28s} {r['score'][-1]:6.3f} {mean:10.3f} "
                  f"{'not confirmed' if hit is None else f'{hit / 1e6:.2f} M':>18s}")
        finals = [r["score"][-1] for r in runs.values()]
        print(f"  {label}: final {np.mean(finals):.3f} (range {min(finals):.3f}-{max(finals):.3f})")
        if label == ref_label:
            continue
        pairs = [(hits[ref_label][s], h) for s, h in hits[label].items() if s in hits[ref_label]]
        done = [(a, b) for a, b in pairs if a is not None and b is not None]
        saving = [a + extra.get(ref_label, 0.0) - b - extra.get(label, 0.0) for a, b in done]
        if done:
            print(f"  {label}: saving vs {ref_label}, median {np.median(saving):.0f} decisions "
                  f"(with extra cost; without {np.median([a - b for a, b in done]):.0f}), "
                  f"faster in {sum(x > 0 for x in saving)}/{len(done)}"
                  + (f", {len(pairs) - len(done)} censored" if len(done) < len(pairs) else ""))
        else:
            print(f"  {label}: saving vs {ref_label} not available, all {len(pairs)} censored")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
