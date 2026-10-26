"""Compare training variants on the same days, matched by training seed (pre-registered analysis).

Reads an evaluation CSV of scripts/evaluate.py whose checkpoints are named run/checkpoint, with
runs named <variant>_s<seed>. For each variant against the reference, on the seeds both have:
  score     mean of the medium and high daily on-time rates; difference averaged over the seeds
            per day, 95 % bootstrap interval over days
  medium, high   the same per load
  seeds     each seed's mean score difference
  served    the same as score for the served rate
  plasma    high-load on-time rate of plasma pooled over days and seeds, bootstrap over days

Usage:
    python scripts/compare_variants.py runs/m3_screen_eval_all.csv --reference m3_set
"""
import argparse
import csv
import re
from collections import defaultdict

import numpy as np

from medical_uav.evaluation import paired_ci

LOADS = ("medium", "high")


def read(path: str) -> dict:
    """{variant: {seed: {load: {day: row}}}} for the checkpoints of runs named <variant>_s<seed>."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for r in csv.DictReader(open(path)):
        m = re.match(r"(.+)_s(\d+)/", r["method"])
        if m:
            data[m.group(1)][int(m.group(2))][r["load"]][int(r["seed"])] = r
    return data


def per_day(runs: dict, seeds, days, value) -> np.ndarray:
    """value(rows by load) for every seed and day, [seeds, days]."""
    return np.array([[value({load: runs[s][load][d] for load in LOADS}) for d in days]
                     for s in seeds])


def plasma_ci(runs, ref, seeds, days, n_boot: int = 10_000) -> tuple[float, float, float]:
    """Difference in high-load pooled plasma on-time rate, bootstrap over days."""
    def counts(group):
        rows = [[group[s]["high"][d] for d in days] for s in seeds]
        return (np.array([[float(r["on_time_count_plasma"]) for r in day] for day in zip(*rows)]),
                np.array([[float(r["released_plasma"]) for r in day] for day in zip(*rows)]))
    (a_on, a_rel), (b_on, b_rel) = counts(runs), counts(ref)
    diff = lambda i: a_on[i].sum() / a_rel[i].sum() - b_on[i].sum() / b_rel[i].sum()
    boot = np.random.default_rng(0).integers(len(days), size=(n_boot, len(days)))
    samples = [diff(i) for i in boot]
    return diff(np.arange(len(days))), np.percentile(samples, 2.5), np.percentile(samples, 97.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv")
    parser.add_argument("--reference", required=True, help="variant name, e.g. m3_set")
    args = parser.parse_args()

    data = read(args.csv)
    ref = data[args.reference]
    score = lambda rows: np.mean([float(rows[load]["on_time_rate"]) for load in LOADS])
    served = lambda rows: np.mean([float(rows[load]["served_rate"]) for load in LOADS])
    fmt = lambda c: f"{c[0] * 100:+5.1f} [{c[1] * 100:+5.1f}, {c[2] * 100:+5.1f}]"
    print(f"differences in points, variant - {args.reference}, 95 % bootstrap intervals over days")
    for name, runs in data.items():
        if name == args.reference:
            continue
        seeds = sorted(set(runs) & set(ref))
        days = sorted(ref[seeds[0]]["high"])
        diff = lambda value: per_day(runs, seeds, days, value) - per_day(ref, seeds, days, value)
        d_score = diff(score)
        print(f"\n{name} (seeds {', '.join(map(str, seeds))})")
        print(f"  score   {fmt(paired_ci(d_score.mean(0)))}")
        for load in LOADS:
            d = diff(lambda rows: float(rows[load]["on_time_rate"]))
            print(f"  {load:7s} {fmt(paired_ci(d.mean(0)))}")
        per_seed = zip(seeds, d_score.mean(1))
        print("  seeds   " + "  ".join(f"s{s} {v * 100:+.1f}" for s, v in per_seed))
        print(f"  served  {fmt(paired_ci(diff(served).mean(0)))}")
        print(f"  plasma  {fmt(plasma_ci(runs, ref, seeds, days))}  (high load, pooled)")


if __name__ == "__main__":
    main()
