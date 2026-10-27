"""Multi-map (B) vs single-map (A): registered endpoints, a two-way bootstrap and final checkpoints.

Run from the repository root after the runs in docs/superpowers/specs/2026-10-27-multimap-preregistration.md.
"""
import csv
from collections import defaultdict

import numpy as np

from medical_uav.evaluation import paired_ci

SEEDS = (11, 22, 33, 44, 55)


def rates(paths, with_map):
    """{(map, load, day): {run: on-time rate}} for checkpoints named <run>/<checkpoint>."""
    out = defaultdict(dict)
    for path in paths:
        for r in csv.DictReader(open(path)):
            if "/" in r["method"]:
                key = (int(r["map"]) if with_map else 0, r["load"], int(r["seed"]))
                out[key][r["method"]] = float(r["on_time_rate"])
    return out


def arm(data, ckpts):
    """{(map, load, day): seed-averaged rate} for one arm's checkpoints (one per seed)."""
    return {k: np.mean([v[c] for c in ckpts]) for k, v in data.items() if all(c in v for c in ckpts)}


def effects(A, B, loads):
    """B - A per (map, day), averaged over the loads."""
    keys = sorted({(m, d) for m, l, d in A})
    return {(m, d): np.mean([B[(m, l, d)] - A[(m, l, d)] for l in loads]) for m, d in keys}


def map_bootstrap(eff):
    maps = sorted({m for m, _ in eff})
    per_map = np.array([np.mean([v for (m, _), v in eff.items() if m == mm]) for mm in maps])
    boot = np.random.default_rng(0).choice(per_map, (10_000, len(maps))).mean(axis=1)
    return per_map.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def two_way_bootstrap(eff):
    """Maps and demand days resampled independently; a sampled day is shared across maps."""
    maps, days = sorted({m for m, _ in eff}), sorted({d for _, d in eff})
    grid = np.array([[eff[(m, d)] for d in days] for m in maps])
    rng = np.random.default_rng(0)
    boot = [grid[np.ix_(rng.integers(len(maps), size=len(maps)),
                        rng.integers(len(days), size=len(days)))].mean() for _ in range(10_000)]
    return grid.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


fmt = lambda c: f"{c[0] * 100:+.2f} [{c[1] * 100:+.2f}, {c[2] * 100:+.2f}]"
sel = {"A": [f"m3_set_s11/ckpt_06000640", "m3_set_s22/ckpt_05500928", "m3_set_s33/ckpt_04501504",
             "m3_set_s44/ckpt_06000640", "m3_set_s55/ckpt_05001216"],
       "B": ["mm_set_s11/ckpt_05001216", "mm_set_s22/ckpt_06000640", "mm_set_s33/ckpt_05500928",
             "mm_set_s44/ckpt_06000640", "mm_set_s55/ckpt_05500928"]}
final = {a: [f"{'m3' if a == 'A' else 'mm'}_set_s{s}/ckpt_06000640" for s in SEEDS] for a in "AB"}
maps_data = rates(["runs/map_stress.csv", "runs/map_stress_mm.csv", "runs/map_stress_finals.csv"], True)
zero_data = rates(["runs/m3_screen_eval_all.csv", "runs/mm_screen_eval.csv",
                   "runs/finals_screen_eval.csv"], False)
for name, ck in (("selected checkpoints", sel), ("final 6 M checkpoints", final)):
    print(f"\n== {name}")
    A, B = arm(maps_data, ck["A"]), arm(maps_data, ck["B"])
    for loads in (("medium", "high"), ("medium",), ("high",)):
        eff = effects(A, B, loads)
        print(f"unseen maps 1-10, {'+'.join(loads):12s} B - A: by maps {fmt(map_bootstrap(eff))}; "
              f"maps and days {fmt(two_way_bootstrap(eff))}")
    A0, B0 = arm(zero_data, ck["A"]), arm(zero_data, ck["B"])
    for loads in (("medium", "high"), ("medium",), ("high",)):
        days = sorted({d for _, l, d in A0 if l == "high"})
        d = [np.mean([B0[(0, l, x)] - A0[(0, l, x)] for l in loads]) for x in days]
        print(f"map 0 screen,     {'+'.join(loads):12s} B - A: {fmt(paired_ci(d))}")
