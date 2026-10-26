"""Why orders fail at medium and high load: per-order outcomes, backlog and the demand's workload.

For every policy and day it records each order (type, route, hour of release, inside a peak or
not, outcome: on time, late, expired before pickup, expired on board) and the backlog (waiting
orders) at each decision. It prints the failure mix, failure rates inside and outside the
peaks, by type and by node, the backlog by hour, a snowball check (does the morning peak's
backlog predict the afternoon's failures?) and the demand's workload against the time one drone
has. Writes orders.csv and days.csv.

Workload of an order: loading and unloading (2 x handling, which no policy can avoid) plus the
loaded flight from origin to destination (which batched boxes can share); flights to the
origin, detours and charging come on top.

Usage:
    python scripts/failure_analysis.py --checkpoints checkpoints/ppo_set.pt --methods slack \\
        --out outputs/failures
"""
import argparse
import csv
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from medical_uav.env import MedicalUAVEnv
from medical_uav.env.config import EPS, NODE_NAMES, ORDER_TYPES
from medical_uav.evaluation import checkpoint_labels
from medical_uav.splits import SCREEN_SEEDS

import evaluate  # scripts/evaluate.py: the same policies as the evaluation

HOURS = 9  # 08:00-17:00


def outcome(o) -> str:
    if o.status == "delivered":
        return "on time" if o.end_time - o.release <= o.limit + EPS else "late"
    if o.status == "expired":
        return "expired on board" if o.pickup_time is not None else "expired before pickup"
    return "open"


def run_job(job):
    """One day of one policy: per-order records and the backlog at each decision."""
    torch.set_num_threads(1)
    method, checkpoint, label, load, seed = job
    policy = next(iter(evaluate.make_policies([method] if method else [],
                                              [checkpoint] if checkpoint else []).values()))
    env = MedicalUAVEnv(load=load)
    obs, info = env.reset(options={"demand_seed": seed})
    backlog, done = [], False
    while not done:
        backlog.append((env.sim.t, len(env.sim.pending)))
        obs, _, done, _, info = env.step(policy(env, obs, info["action_mask"]))
    sim, cfg = env.sim, env.cfg
    peaks = cfg.peak_windows
    orders = [{"method": label, "load": load, "seed": seed, "id": o.id,
               "type": ORDER_TYPES[o.type_idx].name, "origin": NODE_NAMES[o.origin],
               "dest": NODE_NAMES[o.dest], "release": o.release, "hour": int(o.release // 60),
               "peak": any(a <= o.release < b for a, b in peaks), "outcome": outcome(o),
               "work": 2 * cfg.handling_min + sim.travel_time(o.origin, o.dest)}
              for o in sim.orders]
    morning = max([n for t, n in backlog if peaks[0][0] <= t < peaks[0][1]], default=0)
    afternoon = [o for o in orders if o["release"] >= peaks[0][1] + 60]
    day = {"method": label, "load": load, "seed": seed, "orders": len(orders),
           "on_time": sum(o["outcome"] == "on time" for o in orders),
           "morning_backlog": morning, "afternoon_orders": len(afternoon),
           "afternoon_failures": (sum(o["outcome"] != "on time" for o in afternoon)
                                  / len(afternoon) if afternoon else np.nan),
           **{f"backlog_{8 + h}h": max([n for t, n in backlog if h * 60 <= t < (h + 1) * 60],
                                      default=0) for h in range(HOURS)}}
    return orders, day


def share(rows, cond) -> str:
    n = len(rows)
    k = sum(1 for r in rows if cond(r))
    return f"{100 * k / n:5.1f}%" if n else "  -  "


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--methods", default="slack")
    parser.add_argument("--checkpoints", nargs="*", default=[])
    parser.add_argument("--loads", default="medium,high")
    parser.add_argument("--days", type=int, default=len(SCREEN_SEEDS))
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--out", default="outputs/failures")
    args = parser.parse_args()

    named = [(m, None, m) for m in args.methods.split(",") if m]
    named += [(None, c, label) for c, label in
              zip(args.checkpoints, checkpoint_labels(args.checkpoints))]
    seeds = list(SCREEN_SEEDS)[:args.days]
    loads = args.loads.split(",")
    jobs = [(m, c, label, load, s) for m, c, label in named for load in loads for s in seeds]
    with Pool(args.processes) as pool:
        results = pool.map(run_job, jobs, chunksize=1)
    orders = [o for os_, _ in results for o in os_]
    days = [d for _, d in results]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("orders.csv", orders), ("days.csv", days)):
        with open(out / name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    labels = [label for _, _, label in named]
    fail = lambda r: r["outcome"] != "on time"
    for load in loads:
        print(f"\n===== {load} load, {len(seeds)} days =====")
        mine = [o for o in orders if o["load"] == load and o["method"] == labels[0]]
        # workload of the demand, the same for every policy
        per_hour = defaultdict(float)
        for o in mine:
            per_hour[o["hour"]] += o["work"] / len(seeds)
        print("demand workload, drone-minutes of handling and loaded flight per hour of release "
              "(60 available, before charging):")
        print("  " + "  ".join(f"{8 + h}h {per_hour[h]:5.1f}" for h in range(8)))
        for label in labels:
            rows = [o for o in orders if o["load"] == load and o["method"] == label]
            ds = [d for d in days if d["load"] == load and d["method"] == label]
            print(f"\n{label}: on time {share(rows, lambda r: r['outcome'] == 'on time')}")
            where = lambda key, value: [r for r in rows if r[key] == value]
            kinds = ("late", "expired before pickup", "expired on board")
            print("  failures: " + ", ".join(
                f"{k} {share(rows, lambda r, k=k: r['outcome'] == k)}" for k in kinds))
            print(f"  failure rate in peaks {share(where('peak', True), fail)}, "
                  f"outside {share(where('peak', False), fail)}")
            print("  by type:   " + "  ".join(
                f"{t.name} {share(where('type', t.name), fail)}" for t in ORDER_TYPES))
            print("  by origin: " + "  ".join(
                f"{n} {share(where('origin', n), fail)}" for n in NODE_NAMES[:6]))
            print("  by hour:   " + "  ".join(
                f"{8 + h}h {share(where('hour', h), fail)}" for h in range(8)))
            print("  backlog (mean of the hour's maximum): " + "  ".join(
                f"{8 + h}h {np.mean([d[f'backlog_{8 + h}h'] for d in ds]):4.1f}"
                for h in range(HOURS)))
            # snowball check: does the morning peak's backlog raise the failure rate of orders
            # released from 12:00, beyond what the afternoon's own number of orders explains?
            mb = np.array([d["morning_backlog"] for d in ds], float)
            ao = np.array([d["afternoon_orders"] for d in ds], float)
            af = np.array([d["afternoon_failures"] for d in ds], float)
            ok = ~np.isnan(af)
            if ok.sum() > 3:
                X = np.column_stack([np.ones(ok.sum()), mb[ok], ao[ok]])
                coef = np.linalg.lstsq(X, af[ok], rcond=None)[0]
                print(f"  snowball check: each extra order in the morning backlog changes the "
                      f"afternoon failure rate by {coef[1] * 100:+.2f} points (afternoon orders "
                      f"held fixed; each extra afternoon order: {coef[2] * 100:+.2f})")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
