"""Calibration for gate G1: load levels against the baselines, map statistics, throughput.

Examples:
    python scripts/calibrate.py                                   # λ grid on the dev seeds
    python scripts/calibrate.py --tune-theta --map-check 200 --benchmark
"""
import argparse
import csv
import os
import time

import numpy as np

from medical_uav.baselines.heuristics import edf_policy, random_policy, slack_aware_policy
from medical_uav.env import EnvConfig, MedicalUAVEnv
from medical_uav.env.config import EPS
from medical_uav.env.map_gen import generate_map
from medical_uav.env.observation import direct_completion_time
from medical_uav.metrics import episode_metrics
from medical_uav.splits import DEV_SEEDS

DEFAULT_LAMBDAS = [0.04, 0.06, 0.08, 0.10, 0.12, 0.14]
THETAS = [20.0, 30.0, 40.0]


def run_day(env, policy, demand_seed, lam):
    """One day; also tracks pool size and hidden on-time-feasible orders at every decision."""
    env.reset(options={"demand_seed": demand_seed, "lam": lam})
    pool_sizes, hidden_counts, done = [], [], False
    while not done:
        sim = env.sim
        feasible = sum(1 for o in sim.pending
                       if sim.t + direct_completion_time(sim, o) <= o.deadline + EPS)
        visible_feasible = sum(1 for o in env.slots
                               if sim.t + direct_completion_time(sim, o) <= o.deadline + EPS)
        pool_sizes.append(len(sim.pending))
        hidden_counts.append(feasible - visible_feasible)
        _, _, done, _, _ = env.step(policy(env))
    m = episode_metrics(env.sim)
    m["km_per_delivery"] = m["km_flown"] / max(1, m["delivered"])
    m["pool_sizes"] = pool_sizes
    m["hidden_counts"] = hidden_counts
    return m


def summarize(results, lam, name):
    pools = np.concatenate([r["pool_sizes"] for r in results])
    hidden = np.concatenate([r["hidden_counts"] for r in results])
    row = {"lambda": lam, "policy": name, "days": len(results), "decisions": len(pools),
           "orders": np.mean([r["n_orders"] for r in results])}
    for key in ["on_time_rate", "served_rate", "expiry_rate", "rejected", "km_per_delivery",
                "charge_share", "busy_share"]:
        values = [r[key] for r in results]
        row[key] = float(np.mean(values))
        row[key + "_std"] = float(np.std(values))
    row["pool_mean"] = float(pools.mean())
    row["pool_p95"] = float(np.percentile(pools, 95))
    row["pool_max"] = int(pools.max())
    row["hidden_pct"] = 100.0 * float(np.mean(hidden > 0))  # % of decisions hiding a feasible order
    row["hidden_mean"] = float(hidden.mean())
    return row


def map_check(n_maps, cfg):
    ok, tries = 0, []
    for seed in range(n_maps):
        try:
            tries.append(generate_map(seed, cfg).tries)
            ok += 1
        except RuntimeError:
            pass
    print(f"maps: {ok}/{n_maps} generated, mean attempts {np.mean(tries):.2f}")
    m = generate_map(cfg.map_seed, cfg)
    straight = np.linalg.norm(m.node_xy[:, None] - m.node_xy[None, :], axis=2)
    ratio = np.max(np.where(straight > 0, m.dist_km / np.maximum(straight, EPS), 1.0))
    print(f"map_seed {cfg.map_seed}: max A*/straight-line ratio {ratio:.3f}")


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


def tune_theta(env, seeds, lam):
    print(f"Slack-aware theta on dev seeds at λ={lam}:")
    for theta in THETAS:
        policy = lambda e, th=theta: slack_aware_policy(e, th)  # noqa: E731
        rate = np.mean([run_day(env, policy, s, lam)["on_time_rate"] for s in seeds])
        print(f"  theta {theta:4.0f}: on-time {rate:.3f}")


def write_tables(rows, out_dir, cfg, seeds):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "calibration.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| λ | policy | orders | on-time | served | expired | rejected | km/deliv | charge | "
             "busy | pool mean/p95/max | hidden % | hidden mean |",
             "|" + "---|" * 13]
    for r in rows:
        lines.append(
            f"| {r['lambda']:.2f} | {r['policy']} | {r['orders']:.1f} | "
            f"{r['on_time_rate']:.2f} ± {r['on_time_rate_std']:.2f} | {r['served_rate']:.2f} | "
            f"{r['expiry_rate']:.2f} | {r['rejected']:.1f} | {r['km_per_delivery']:.2f} | "
            f"{r['charge_share']:.2f} | {r['busy_share']:.2f} | "
            f"{r['pool_mean']:.1f}/{r['pool_p95']:.0f}/{r['pool_max']} | "
            f"{r['hidden_pct']:.1f} | {r['hidden_mean']:.2f} |")
    header = (f"map_seed = {cfg.map_seed}; dev seeds {seeds[0]}..{seeds[-1]} "
              f"({len(seeds)} days per row); {rows[0]['decisions']} decisions in the first row\n\n")
    table = "\n".join(lines)
    with open(os.path.join(out_dir, "calibration.md"), "w") as f:
        f.write(header + table + "\n")
    print(table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambdas", type=float, nargs="*", default=DEFAULT_LAMBDAS)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(DEV_SEEDS))
    parser.add_argument("--theta", type=float, default=20.0,
                        help="Slack-aware charging threshold")
    parser.add_argument("--tune-theta", action="store_true")
    parser.add_argument("--map-check", type=int, default=0)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--out", default="outputs/calibration")
    args = parser.parse_args()
    cfg = EnvConfig()

    if args.map_check:
        map_check(args.map_check, cfg)
    if args.benchmark:
        benchmark(cfg)
    env = MedicalUAVEnv(config=cfg)
    if args.tune_theta:
        tune_theta(env, args.seeds, cfg.loads["medium"])
    if not args.lambdas:
        return

    rng = np.random.default_rng(0)
    policies = {"slack": lambda e: slack_aware_policy(e, args.theta), "edf": edf_policy,
                "random": lambda e: random_policy(e, rng)}
    rows = []
    for lam in args.lambdas:
        for name, policy in policies.items():
            rows.append(summarize([run_day(env, policy, s, lam) for s in args.seeds], lam, name))
    write_tables(rows, args.out, cfg, args.seeds)


if __name__ == "__main__":
    main()
