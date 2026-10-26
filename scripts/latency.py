"""Decision time of policies, one decision at a time on one CPU thread.

Each method serves whole selection days at each load until it has taken at least --decisions
decisions; this is repeated --passes times. Every decision's time is saved; the summary gives
the mean, median, 95th percentile and maximum per method, load and pass.

Usage:
    python scripts/latency.py --methods scenario4,scenario16 --decisions 1000 --passes 3 \\
        --out runs/latency.csv
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from medical_uav.env import MedicalUAVEnv
from medical_uav.splits import SELECT_SEEDS

import evaluate  # scripts/evaluate.py: the same policies as the evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--methods", default="slack,rollout")
    parser.add_argument("--checkpoints", nargs="*", default=[])
    parser.add_argument("--loads", default="low,medium,high")
    parser.add_argument("--decisions", type=int, default=1000)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--out", default="outputs/latency.csv")
    args = parser.parse_args()
    torch.set_num_threads(1)

    policies = evaluate.make_policies([m for m in args.methods.split(",") if m], args.checkpoints)
    rows = []
    for name, policy in policies.items():
        for load in args.loads.split(","):
            for p in range(1, args.passes + 1):
                times, days = [], iter(SELECT_SEEDS)
                while len(times) < args.decisions:
                    seed = next(days)
                    env = MedicalUAVEnv(load=load)
                    obs, info = env.reset(options={"demand_seed": seed})
                    done = False
                    while not done:
                        t0 = time.perf_counter()
                        a = policy(env, obs, info["action_mask"])
                        times.append(time.perf_counter() - t0)
                        rows.append({"method": name, "load": load, "pass": p, "seed": seed,
                                     "ms": times[-1] * 1000})
                        obs, _, done, _, info = env.step(a)
                ms = np.array(times) * 1000
                print(f"{name:16s} {load:6s} pass {p}: {len(ms)} decisions  mean {ms.mean():7.2f}  "
                      f"p50 {np.median(ms):7.2f}  p95 {np.percentile(ms, 95):7.2f}  "
                      f"max {ms.max():7.2f} ms", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
