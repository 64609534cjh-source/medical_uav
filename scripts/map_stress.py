"""Policies on maps they were never trained on (pre-registered stress test, no retraining).

Every policy serves the same days (the first 20 selection days) on each of the given maps at
medium and high load. The summary gives each method's mean on-time rate, and for the trained
networks (seeds averaged per day) their difference to Slack-aware on each map, with a 95 %
bootstrap over maps (the map is the unit: ten maps give limited precision).

Usage:
    python scripts/map_stress.py --checkpoints runs/m3_set_s11/ckpt_06000640.pt --maps 1-10
"""
import argparse
import csv
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from medical_uav.env import EnvConfig
from medical_uav.evaluation import checkpoint_labels, run_day
from medical_uav.splits import SELECT_SEEDS

import evaluate  # scripts/evaluate.py: the same policies as the evaluation


def run_job(job) -> dict:
    """One day of one policy on one map."""
    torch.set_num_threads(1)
    method, checkpoint, label, map_seed, load, seed = job
    policy = next(iter(evaluate.make_policies([method] if method else [],
                                              [checkpoint] if checkpoint else []).values()))
    row = run_day(policy, load, seed, EnvConfig(map_seed=map_seed))
    return {"method": label, "map": map_seed, "load": load, "seed": seed, **row}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--methods", default="slack,rollout")
    parser.add_argument("--checkpoints", nargs="*", default=[])
    parser.add_argument("--maps", default="1-10", help="first-last map seed")
    parser.add_argument("--loads", default="medium,high")
    parser.add_argument("--days", type=int, default=20)
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--out", default="outputs/map_stress.csv")
    args = parser.parse_args()

    first, last = (int(x) for x in args.maps.split("-"))
    maps, loads = list(range(first, last + 1)), args.loads.split(",")
    seeds = list(SELECT_SEEDS)[:args.days]
    named = [(m, None, m) for m in args.methods.split(",") if m]
    named += [(None, c, label) for c, label in
              zip(args.checkpoints, checkpoint_labels(args.checkpoints))]
    jobs = [(m, c, label, map_seed, load, s) for m, c, label in named for map_seed in maps
            for load in loads for s in seeds]
    with Pool(args.processes) as pool:
        rows = pool.map(run_job, jobs, chunksize=1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    rate = defaultdict(dict)  # (method, map, load) -> {seed: on-time rate}
    for r in rows:
        rate[r["method"], r["map"], r["load"]][r["seed"]] = r["on_time_rate"]
    labels = [label for _, _, label in named]
    networks = checkpoint_labels(args.checkpoints)
    for load in loads:
        print(f"\n{load} load, mean on-time over {len(maps)} maps x {len(seeds)} days")
        for label in labels:
            print(f"  {label:28s} {np.mean([v for m in maps for v in rate[label, m, load].values()]):.3f}")
        if not networks:
            continue
        # per map: networks averaged per day, minus Slack-aware, averaged over the days
        diff = np.array([np.mean([np.mean([rate[n, m, load][s] for n in networks])
                                  - rate["slack", m, load][s] for s in seeds]) for m in maps])
        boot = np.random.default_rng(0).choice(diff, (10_000, len(diff))).mean(axis=1)
        print(f"  networks - slack: {diff.mean() * 100:+.1f} points "
              f"[{np.percentile(boot, 2.5) * 100:+.1f}, {np.percentile(boot, 97.5) * 100:+.1f}] "
              f"(bootstrap over maps); per map: "
              + " ".join(f"{m}:{d * 100:+.0f}" for m, d in zip(maps, diff)))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
