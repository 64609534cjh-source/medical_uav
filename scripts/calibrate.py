"""Calibrate load levels against the baselines and report map statistics and throughput.

Usage:
    python scripts/calibrate.py
    python scripts/calibrate.py --tune-theta --map-check 200 --benchmark
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

from medical_uav.baselines.heuristics import (DEFAULT_THETA, edf_policy, random_policy,
                                              slack_aware_policy)
from medical_uav.env import EnvConfig, MedicalUAVEnv
from medical_uav.env.config import EPS
from medical_uav.env.map_gen import generate_map
from medical_uav.env.observation import direct_completion_time
from medical_uav.metrics import episode_metrics
from medical_uav.splits import DEV_SEEDS

DEFAULT_LAMBDAS = [0.04, 0.06, 0.09, 0.12, 0.14]  # base rates, orders per minute
THETAS = [20.0, 30.0, 40.0]  # charging thresholds tried by --tune-theta
TUNABLE = {"edf": edf_policy, "slack": slack_aware_policy}  # heuristics with a threshold
DEFAULT_THETAS = {name: DEFAULT_THETA for name in TUNABLE}


def run_day(env, policy, demand_seed, lam):
    """Run one day and return its metrics plus, at every decision, the pending-pool size and
    the number of orders that could be on time but are not shown in the slots."""
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
    """Return one table row of mean and std metrics over the given days."""
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
    # % of decisions where an order that could still be on time is not shown in the slots
    row["hidden_pct"] = 100.0 * float(np.mean(hidden > 0))
    row["hidden_mean"] = float(hidden.mean())
    return row


def map_check(n_maps, cfg):
    """Print how many map seeds generate and how much no-fly zones lengthen routes."""
    ok, tries = 0, []
    for seed in range(n_maps):
        try:
            tries.append(generate_map(seed, cfg).tries)
            ok += 1
        except RuntimeError:
            pass
    print(f"maps: {ok}/{n_maps} generated, mean node layouts {np.mean(tries):.2f}")
    m = generate_map(cfg.map_seed, cfg)
    straight = np.linalg.norm(m.node_xy[:, None] - m.node_xy[None, :], axis=2)
    ratio = np.max(np.where(straight > 0, m.dist_km / np.maximum(straight, EPS), 1.0))
    print(f"map_seed {cfg.map_seed}: max A*/straight-line ratio {ratio:.3f}")


def benchmark(cfg, episodes=20):
    """Print simulation speed in decisions per second."""
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


def make_policy(name: str, demand_seed: int, thetas: dict):
    """Return the policy for one day; Random is seeded per day so results ignore run order."""
    if name == "random":
        rng = np.random.default_rng(demand_seed)
        return lambda e: random_policy(e, rng)
    fn, theta = TUNABLE[name], thetas[name]
    return lambda e: fn(e, theta)


def select_thetas(results: dict) -> dict:
    """Return the best charging threshold per heuristic from tune_thetas results."""
    return {name: max(by_theta, key=by_theta.get) for name, by_theta in results.items()}


def tune_thetas(env, seeds, lam, thetas=THETAS) -> dict:
    """Return the mean on-time rate for each heuristic and threshold as {name: {theta: rate}}."""
    results = {}
    for name, policy_fn in TUNABLE.items():
        results[name] = {}
        for theta in thetas:
            policy = lambda e, fn=policy_fn, th=theta: fn(e, th)  # noqa: E731
            rates = [run_day(env, policy, s, lam)["on_time_rate"] for s in seeds]
            results[name][theta] = float(np.mean(rates))
    return results


POLICY_NAMES = {"slack": "Slack-aware", "edf": "EDF", "random": "Random"}


def describe_seeds(seeds) -> str:
    """Return the seeds as a range when they are consecutive, else as a list."""
    seeds = list(seeds)
    if seeds == list(range(seeds[0], seeds[0] + len(seeds))):
        return f"{seeds[0]}-{seeds[-1]}" if len(seeds) > 1 else str(seeds[0])
    return ", ".join(map(str, seeds))


def settings_line(cfg, seeds, thetas, tuned: bool) -> str:
    """Return the settings every table row was computed with."""
    return (f"map seed {cfg.map_seed}; demand seeds {describe_seeds(seeds)} ({len(seeds)} days "
            f"per row); charging threshold EDF {thetas['edf']:.0f}%, Slack-aware "
            f"{thetas['slack']:.0f}% ({'tuned in this run' if tuned else 'defaults'})")


def print_table(rows) -> None:
    """Print the main columns: on-time rate and expiry rate per base rate and policy."""
    print(f"{'rate':>6s}  {'policy':12s} {'orders/day':>10s}  {'on time, mean (SD)':>19s}  "
          f"{'expired, mean':>13s}")
    for r in rows:
        on_time = f"{100 * r['on_time_rate']:.1f}% ({100 * r['on_time_rate_std']:.1f} pp)"
        print(f"{r['lambda']:6.2f}  {POLICY_NAMES[r['policy']]:12s} {r['orders']:10.1f}  "
              f"{on_time:>19s}  {100 * r['expiry_rate']:12.1f}%")


def write_tables(rows, out_dir, cfg, seeds, thetas, tuned=False):
    """Write the calibration results as CSV and Markdown tables."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "calibration.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| λ | policy | orders | on-time (mean ± day SD) | served | expired | rejected | "
             "km/deliv | charge | busy | decisions | pool mean/p95/max | hidden % | hidden mean |",
             "|" + "---|" * 14]
    for r in rows:
        lines.append(
            f"| {r['lambda']:.2f} | {r['policy']} | {r['orders']:.1f} | "
            f"{r['on_time_rate']:.2f} ± {r['on_time_rate_std']:.2f} | {r['served_rate']:.2f} | "
            f"{r['expiry_rate']:.2f} | {r['rejected']:.1f} | {r['km_per_delivery']:.2f} | "
            f"{r['charge_share']:.2f} | {r['busy_share']:.2f} | {r['decisions']} | "
            f"{r['pool_mean']:.1f}/{r['pool_p95']:.0f}/{r['pool_max']} | "
            f"{r['hidden_pct']:.1f} | {r['hidden_mean']:.2f} |")
    header = (f"Command: python scripts/calibrate.py {' '.join(sys.argv[1:])}\n\n"
              f"{settings_line(cfg, seeds, thetas, tuned)}.\n\n"
              "λ is the base arrival rate in orders per minute. Rates are means of the daily "
              "rates over the days of a row, with the population SD across days; a day's "
              "on-time rate is on-time deliveries / all orders released that day (rejected "
              "included). Pool statistics are pooled over all decisions; hidden % is the share "
              "of decisions at which an order that could still be on time (direct estimate) is "
              "not among the 12 visible slots.\n\n")
    with open(os.path.join(out_dir, "calibration.md"), "w") as f:
        f.write(header + "\n".join(lines) + "\n")
    print(f"Saved: {os.path.join(out_dir, 'calibration.csv')}")
    print(f"Saved: {os.path.join(out_dir, 'calibration.md')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambdas", type=float, nargs="*", default=DEFAULT_LAMBDAS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEV_SEEDS))
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
    thetas = dict(DEFAULT_THETAS)
    if args.tune_theta:
        lam = cfg.loads["medium"]
        results = tune_thetas(env, args.seeds, lam)
        thetas = select_thetas(results)
        for name, by_theta in results.items():
            rates = ", ".join(f"θ {th:.0f}: {r:.3f}" for th, r in by_theta.items())
            print(f"{name} on dev seeds at λ={lam}: {rates} -> θ {thetas[name]:.0f} used below")
    if not args.lambdas:
        return

    print("\nCalibration")
    print(settings_line(cfg, args.seeds, thetas, args.tune_theta))
    rows = []
    for lam in args.lambdas:
        for name in ["slack", "edf", "random"]:
            days = [run_day(env, make_policy(name, s, thetas), s, lam) for s in args.seeds]
            rows.append(summarize(days, lam, name))
    print_table(rows)
    write_tables(rows, args.out, cfg, args.seeds, thetas, args.tune_theta)


if __name__ == "__main__":
    main()
