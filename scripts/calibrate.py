"""Calibration for gate G1: load levels, map choice and environment throughput.

Examples:
    python scripts/calibrate.py --map-check 200 --lambdas          # map statistics only
    python scripts/calibrate.py --benchmark                        # full λ grid + speed
"""
import argparse
import csv
import os
import time

import numpy as np

from medical_uav.baselines.heuristics import edf_policy, random_policy
from medical_uav.env import EnvConfig, MedicalUAVEnv
from medical_uav.env.config import EPS
from medical_uav.env.map_gen import generate_map
from medical_uav.env.observation import direct_completion_time
from medical_uav.metrics import episode_metrics

DEFAULT_LAMBDAS = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12]
DEV_SEEDS = list(range(100000, 100020))
COLUMNS = ["on_time_rate", "served_rate", "expiry_rate", "rejected", "km_per_delivery",
           "charge_share", "busy_share", "mean_pool", "hidden_feasible"]


def run_with_stats(env, policy, demand_seed, lam):
    """One episode; also tracks pool size and hidden on-time-feasible orders per decision."""
    env.reset(options={"demand_seed": demand_seed, "lam": lam})
    pool_sizes, hidden, done = [], 0, False
    while not done:
        sim = env.sim
        pool_sizes.append(len(sim.pending))
        feasible = [o for o in sim.pending
                    if sim.t + direct_completion_time(sim, o) <= o.deadline + EPS]
        hidden += max(0, len(feasible) - len(env.slots))
        _, _, done, _, _ = env.step(policy(env))
    m = episode_metrics(env.sim)
    delivered = max(1, round(m["served_rate"] * m["n_orders"]))
    m["km_per_delivery"] = m["km_flown"] / delivered
    m["mean_pool"] = float(np.mean(pool_sizes))
    m["hidden_feasible"] = hidden
    return m


def map_check(n_maps, cfg):
    ok, tries = 0, []
    for seed in range(n_maps):
        try:
            tries.append(generate_map(seed, cfg).tries)
            ok += 1
        except RuntimeError:
            pass
    print(f"maps: {ok}/{n_maps} generated, mean attempts {np.mean(tries):.2f}")
    print("seed  max A*/straight ratio")
    for seed in range(20):
        m = generate_map(seed, cfg)
        straight = np.linalg.norm(m.node_xy[:, None] - m.node_xy[None, :], axis=2)
        ratio = np.max(np.where(straight > 0, m.dist_km / np.maximum(straight, EPS), 1.0))
        print(f"{seed:4d}  {ratio:.3f}")


def benchmark(cfg, episodes=20):
    env = MedicalUAVEnv(config=cfg, load="medium")
    rng = np.random.default_rng(0)
    steps, start = 0, time.time()
    for seed in range(episodes):
        _, info = env.reset(seed=seed)
        done = False
        while not done:
            _, _, done, _, info = env.step(int(rng.choice(np.flatnonzero(info["action_mask"]))))
            steps += 1
    rate = steps / (time.time() - start)
    print(f"throughput: {rate:.0f} decisions/s (single process, random policy, {steps} steps)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambdas", type=float, nargs="*", default=DEFAULT_LAMBDAS)
    parser.add_argument("--seeds", type=int, nargs="*", default=DEV_SEEDS)
    parser.add_argument("--map-check", type=int, default=0)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--out", default="outputs/calibration")
    args = parser.parse_args()
    cfg = EnvConfig()

    if args.map_check:
        map_check(args.map_check, cfg)
    if args.benchmark:
        benchmark(cfg)
    if not args.lambdas:
        return

    os.makedirs(args.out, exist_ok=True)
    env = MedicalUAVEnv(config=cfg)
    rng = np.random.default_rng(0)
    policies = {"edf": edf_policy, "random": lambda e: random_policy(e, rng)}
    rows = []
    for lam in args.lambdas:
        for name, policy in policies.items():
            results = [run_with_stats(env, policy, s, lam) for s in args.seeds]
            row = {"lambda": lam, "policy": name,
                   "orders": np.mean([r["n_orders"] for r in results])}
            for col in COLUMNS:
                values = [r[col] for r in results]
                row[col] = float(np.mean(values))
                row[col + "_std"] = float(np.std(values))
            rows.append(row)

    with open(os.path.join(args.out, "calibration.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    header = "| λ | policy | orders | on-time | served | expired | km/deliv | charge | busy | pool | hidden |"
    lines = [header, "|" + "---|" * 11]
    for r in rows:
        lines.append(
            f"| {r['lambda']:.2f} | {r['policy']} | {r['orders']:.1f} | "
            f"{r['on_time_rate']:.2f} ± {r['on_time_rate_std']:.2f} | {r['served_rate']:.2f} | "
            f"{r['expiry_rate']:.2f} | {r['km_per_delivery']:.2f} | {r['charge_share']:.2f} | "
            f"{r['busy_share']:.2f} | {r['mean_pool']:.1f} | {r['hidden_feasible']:.1f} |")
    table = "\n".join(lines)
    with open(os.path.join(args.out, "calibration.md"), "w") as f:
        f.write(f"map_seed = {cfg.map_seed}, dev seeds {args.seeds[0]}..{args.seeds[-1]}\n\n{table}\n")
    print(table)


if __name__ == "__main__":
    main()
