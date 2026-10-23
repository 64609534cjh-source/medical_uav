"""Compare policies on the same days: per-day CSV and a per-load summary against Slack-aware.

Usage:
    python scripts/evaluate.py --methods random,edf,slack,fcfs,nearest --loads low,medium,high
    python scripts/evaluate.py --methods slack --checkpoints checkpoints/bc_set.pt --out outputs/eval.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from medical_uav.agents.policy import NetworkPolicy, load_checkpoint
from medical_uav.baselines.heuristics import (edf_policy, fcfs_policy, nearest_policy, random_policy,
                                              slack_aware_policy)
from medical_uav.evaluation import evaluate, heuristic, paired_ci
from medical_uav.splits import SCREEN_SEEDS, SELECT_SEEDS

SEED_SETS = {"screen": SCREEN_SEEDS, "select": SELECT_SEEDS}


class DailyRandom:
    """Random policy seeded with each day's demand seed, as in the M1 calibration."""

    def __init__(self):
        self.sim, self.rng = None, None

    def __call__(self, env) -> int:
        if env.sim is not self.sim:  # a new day has started
            self.sim, self.rng = env.sim, np.random.default_rng(env.demand_seed)
        return random_policy(env, self.rng)


def make_policies(methods: list[str], checkpoints: list[str]) -> dict:
    """Return the named rule-based policies and one network policy per checkpoint file."""
    rules = {"random": DailyRandom(), "edf": edf_policy, "slack": slack_aware_policy,
             "fcfs": fcfs_policy, "nearest": nearest_policy}
    policies = {m: heuristic(rules[m]) for m in methods}
    for path in checkpoints:
        net, _ = load_checkpoint(path)
        policies[Path(path).stem] = NetworkPolicy(net)
    return policies


def print_summary(rows: list[dict], loads: list[str]) -> None:
    """Print the mean on-time rate per method and load, and the paired difference to Slack-aware."""
    methods = list(dict.fromkeys(r["method"] for r in rows))
    for load in loads:
        print(f"\n{load} load")
        by = {m: {r["seed"]: r for r in rows if r["method"] == m and r["load"] == load} for m in methods}
        for m in methods:
            days = by[m]
            line = (f"  {m:12s} on-time {np.mean([d['on_time_rate'] for d in days.values()]):.3f}"
                    f"  served {np.mean([d['served_rate'] for d in days.values()]):.3f}")
            if "slack" in by and m != "slack":
                diffs = [days[s]["on_time_rate"] - by["slack"][s]["on_time_rate"] for s in days]
                mean, lo, hi = paired_ci(np.array(diffs))
                line += f"  vs slack {mean:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--methods", default="random,edf,slack,fcfs,nearest")
    parser.add_argument("--checkpoints", nargs="*", default=[])
    parser.add_argument("--loads", default="low,medium,high")
    parser.add_argument("--seeds", choices=list(SEED_SETS), default="screen")
    parser.add_argument("--out", default="outputs/evaluation.csv")
    args = parser.parse_args()

    loads = args.loads.split(",")
    methods = [m for m in args.methods.split(",") if m]
    rows = evaluate(make_policies(methods, args.checkpoints), loads, SEED_SETS[args.seeds])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print_summary(rows, loads)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
