"""Registered statistics of the locked test, from the per-day files of scripts/locked_test.py.

Writes <out>/map0.csv and <out>/maps.csv (one row per entrant and day, decision times left out)
and <out>/controls.csv (the PPO runs as <run>/selected), and prints:
  - map 0, per load: every entrant's mean daily on-time, served and expired rate;
  - the five-seed PPO (seeds averaged per day) and the shipped model against Slack-aware and both
    planners, with the registered targets; each seed's mean, their SD and range;
  - pooled per-type on-time rates; decision times over all decisions (throughput run);
  - new maps: multi-map minus single-map PPO and both against the rules, bootstrap over maps.
The energy contrasts and the matched-seed controls (including multi-map on map 0) come from
    python scripts/energy_contrasts.py <out>/map0.csv
    python scripts/compare_variants.py <out>/controls.csv --reference m3_set

Usage:
    python scripts/locked_report.py runs/locked
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from medical_uav.env.config import ORDER_TYPES
from medical_uav.evaluation import paired_ci

from locked_test import MULTIMAP, PRIMARY  # scripts/locked_test.py

SHIPPED = "m3_set_s11"
PLANNERS = ("rollout", "scenario16")
# registered targets (roadmap §1): lower bound of PPO - Slack above the margin; planners -2
SLACK_MARGIN = {"low": -0.01, "medium": None, "high": -0.02}
MEDIUM_GAIN = 0.03
PLANNER_MARGIN = -0.02


def load_rows(folder: Path) -> tuple[dict, dict]:
    """{block: [row]}, and the decision times kept apart: {method: [ms, ...]}."""
    blocks, times = defaultdict(list), defaultdict(list)
    for path in sorted(folder.glob("*/*/*.json")):
        row = json.loads(path.read_text())
        times[row["method"]] += row.pop("decision_ms")
        top_ups = row.pop("top_ups")
        row["overrides"] = float("nan") if top_ups is None else top_ups  # energy_contrasts.py
        blocks[row["block"]].append(row)
    return blocks, times


def write_csv(rows: list, path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rate(rows, method, load, key="on_time_rate", m=0) -> dict:
    """{day: value} of one entrant at one load on one map."""
    return {r["seed"]: r[key] for r in rows if r["method"] == method and r["load"] == load
            and r["map"] == m}


def seed_mean(rows, runs, load, m=0) -> dict:
    """{day: on-time averaged over the runs}."""
    per = [rate(rows, run, load, m=m) for run in runs]
    return {d: np.mean([p[d] for p in per]) for d in per[0]}


def fmt(c) -> str:
    return f"{c[0] * 100:+6.2f} [{c[1] * 100:+6.2f}, {c[2] * 100:+6.2f}]"


def against(a: dict, b: dict) -> tuple:
    return paired_ci(np.array([a[d] - b[d] for d in sorted(a)]))


def map0(rows, times) -> None:
    methods = list(dict.fromkeys(r["method"] for r in rows))
    loads = list(dict.fromkeys(r["load"] for r in rows))
    for load in loads:
        n = len(rate(rows, "slack", load))
        print(f"\n== map 0, {load} load, {n} days: mean daily rate (on-time / served / expired)")
        for m in methods:
            v = [np.mean(list(rate(rows, m, load, k).values()))
                 for k in ("on_time_rate", "served_rate", "expiry_rate")]
            print(f"  {m:22s} {v[0]:.3f} / {v[1]:.3f} / {v[2]:.3f}")
        ppo, shipped = seed_mean(rows, PRIMARY, load), rate(rows, SHIPPED, load)
        print("  differences in points, 95 % bootstrap over days:")
        for base in ("slack",) + PLANNERS:
            ref = rate(rows, base, load)
            five, ship = against(ppo, ref), against(shipped, ref)
            if base == "slack":
                target = (f"target ≥ +{MEDIUM_GAIN * 100:.0f}, lower bound > 0" if load == "medium"
                          else f"margin {SLACK_MARGIN[load] * 100:+.0f}")
                met = (five[0] >= MEDIUM_GAIN and five[1] > 0) if load == "medium" \
                    else five[1] > SLACK_MARGIN[load]
            else:
                target, met = f"margin {PLANNER_MARGIN * 100:+.0f}", five[1] > PLANNER_MARGIN
            print(f"  PPO (5 seeds) - {base:10s} {fmt(five)}  {target}: "
                  f"{'met' if met else 'NOT met'}   shipped - {base}: {fmt(ship)}")
        means = [np.mean(list(rate(rows, run, load).values())) for run in PRIMARY]
        print("  seed means " + " ".join(f"{r[-3:]} {v:.3f}" for r, v in zip(PRIMARY, means))
              + f"; SD {np.std(means, ddof=1) * 100:.2f} points, range "
              f"{(max(means) - min(means)) * 100:.2f}")
        print("  pooled on-time by type (on time / released):")
        for m in ["slack", *PLANNERS, SHIPPED]:
            mine = [r for r in rows if r["method"] == m and r["load"] == load]
            pooled = [sum(r[f"on_time_count_{t.name}"] for r in mine)
                      / max(1, sum(r[f"released_{t.name}"] for r in mine)) for t in ORDER_TYPES]
            print(f"    {m:12s} "
                  + "  ".join(f"{t.name} {v:.3f}" for t, v in zip(ORDER_TYPES, pooled)))
    print("\n== decision times over all decisions, ms (a parallel run, not a latency benchmark)")
    for m in methods:
        t = np.array(times[m])
        print(f"  {m:22s} mean {t.mean():8.2f}  p95 {np.percentile(t, 95):8.2f}"
              f"  max {t.max():8.1f}")


def map_bootstrap(per_map: np.ndarray) -> tuple:
    boot = np.random.default_rng(0).choice(per_map, (10_000, len(per_map))).mean(axis=1)
    return per_map.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def new_maps(rows) -> None:
    maps = sorted({r["map"] for r in rows})
    days = sorted({r["seed"] for r in rows})
    print(f"\n== new maps {maps[0]}-{maps[-1]}, {len(days)} days each: differences in points, "
          "seeds averaged per day, 95 % bootstrap over maps")
    for loads in (("medium", "high"), ("medium",), ("high",)):
        arms = {"multi-map": MULTIMAP, "single-map": PRIMARY}
        score = {name: {m: np.mean([np.mean(list(seed_mean(rows, runs, x, m).values()))
                                    for x in loads]) for m in maps} for name, runs in arms.items()}
        for base in ("slack", "rollout"):
            score[base] = {m: np.mean([np.mean(list(rate(rows, base, x, m=m).values()))
                                       for x in loads]) for m in maps}
        print(f"  {'+'.join(loads)}")
        for a, b in (("multi-map", "single-map"), ("single-map", "slack"), ("multi-map", "slack"),
                     ("single-map", "rollout"), ("multi-map", "rollout")):
            diff = np.array([score[a][m] - score[b][m] for m in maps])
            print(f"    {a} - {b:10s} {fmt(map_bootstrap(diff))}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out")
    args = parser.parse_args()
    out = Path(args.out)
    blocks, times = load_rows(out)
    write_csv(blocks["map0"], out / "map0.csv")
    write_csv(blocks["maps"], out / "maps.csv")
    write_csv([{**r, "method": f"{r['method']}/selected"} for r in blocks["map0"]
               if r["method"].startswith(("m3_", "mm_")) and "/" not in r["method"]],
              out / "controls.csv")
    map0(blocks["map0"], times)
    new_maps(blocks["maps"])


if __name__ == "__main__":
    main()
