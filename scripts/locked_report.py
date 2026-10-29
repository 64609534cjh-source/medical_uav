"""Registered statistics of the final test, from the per-day files of scripts/locked_test.py.

Writes <out>/map0.csv and <out>/maps.csv (one row per entrant and day, decision times left out)
and <out>/summary.json (every printed number), and prints, per block and load:
  - every entrant's mean daily on-time, served and expired rate, each PPO group's per-seed means;
  - primary: the final agent against Slack-aware and the rolling horizon, with the targets;
  - secondary: the final agent against every other rule and planner;
  - the development path (map and reward effects) and the controls, with matched seeds;
  - pooled per-type on-time rates; decision times over all decisions (throughput run);
  - operations: lateness (pooled over late deliveries), energy, distance, empty distance, backlog;
  - on the unseen maps, per-map differences of the primary comparisons and maps won;
  - the transfer gap (unseen maps minus map 0, descriptive).
Map 0 is resampled by day, the unseen maps by map (each map's days averaged first); PPO groups
are averaged over their seeds per day. Intervals are per-comparison 95 % percentile intervals.
The energy contrasts come from
    python scripts/energy_contrasts.py <out>/map0.csv

Usage:
    python scripts/locked_report.py runs/final_test | tee runs/final_test/report.txt
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from medical_uav.env.config import ORDER_TYPES
from medical_uav.evaluation import paired_ci

# scripts/locked_test.py
from locked_test import CONTROLS, FINAL, LOADS, MANIFEST, PATH, RULES, expected, sha256

GROUPS = {"final": FINAL, "final3": FINAL[:3], **{g: [r for r in PATH + CONTROLS if r.rsplit("_s", 1)[0] == g]
                             for g in dict.fromkeys(r.rsplit("_s", 1)[0] for r in PATH + CONTROLS)}}
SHIPPED = FINAL[0]
# roadmap §1 targets: medium gain over Slack-aware and margins for the lower bound
MEDIUM_GAIN = 0.03
SLACK_MARGIN = {"low": -0.01, "high": -0.02}
PLANNER_MARGIN = -0.02
EQUIVALENCE = 0.01  # practically equivalent: the interval within ±1 point
OPS = ("energy_used", "km_flown", "empty_km", "mean_backlog")  # per-day means
OPS_METHODS = ["final", "slack", "insertion", "rollout", "scenario16", "horizon"]
PRIMARY = ("slack", "horizon")
OTHERS = ("random", "edf", "fcfs", "nearest", "insertion", "rollout", "scenario16")
# (name, a, b, metric): development path and controls
CONTRASTS = [("map effect, expiry -0.5", "final", "f12_sm_e05", "on_time_rate"),
             ("map effect, v2", "f12_mm_v2", "f12_sm_v2", "on_time_rate"),
             ("reward effect, multi-map", "final", "f12_mm_v2", "on_time_rate"),
             ("reward effect, multi-map (expired)", "final", "f12_mm_v2", "expiry_rate"),
             ("reward effect, single-map", "f12_sm_e05", "f12_sm_v2", "on_time_rate"),
             ("reward effect, single-map (expired)", "f12_sm_e05", "f12_sm_v2", "expiry_rate"),
             ("set - MLP, 12 M each", "f12_sm_v2", "f12_sm_v2_mlp", "on_time_rate"),
             ("set 12 M - MLP 24 M", "f12_sm_v2", "f24_sm_v2_mlp", "on_time_rate"),
             ("final - MLP, multi-map 12 M each", "final3", "f12_mm_e05_mlp", "on_time_rate"),
             ("final - MLP, multi-map 24 M", "final3", "f24_mm_e05_mlp", "on_time_rate"),
             ("BC -> PPO - scratch", "f12_sm_e05_bc", "f12_sm_e05", "on_time_rate"),
             ("BC -> PPO - its imitation start", "f12_sm_e05_bc", "bc_set", "on_time_rate"),
             ("v2 - M1 reward", "f12_sm_v2", "f12_sm_m1", "on_time_rate"),
             ("v2 - on-time-only reward", "f12_sm_v2", "f12_sm_ontime", "on_time_rate")]


def load_rows(folder: Path, days: str, limit: int | None) -> tuple[dict, dict]:
    """{block: [row]}, and the decision times kept apart: {(block, method, load): [ms, ...]}.
    The folder must hold exactly the campaign's days, each file the day its name says, all from
    one campaign of the current manifest with the requested days and limit."""
    jobs = expected(json.loads(MANIFEST.read_text()), days, limit, folder)
    have = set(folder.glob("*/*/*.json"))
    if {job[6] for job in jobs} != have:
        raise SystemExit(f"{len(jobs) - len(have & {j[6] for j in jobs})} days missing, "
                         f"{len(have - {j[6] for j in jobs})} not expected")
    blocks, times, campaigns = defaultdict(list), defaultdict(list), set()
    for job in jobs:
        row = json.loads(job[6].read_text())
        if (row["block"], row["method"], row["map"], row["load"], row["seed"]) != \
                (job[0], job[1], job[3], job[4], job[5]):
            raise SystemExit(f"{job[6]} holds another day")
        campaigns.add(json.dumps(row.pop("campaign"), sort_keys=True))
        times[row["block"], row["method"], row["load"]] += row.pop("decision_ms")
        top_ups = row.pop("top_ups")
        row["overrides"] = float("nan") if top_ups is None else top_ups  # energy_contrasts.py
        blocks[row["block"]].append(row)
    if len(campaigns) != 1:
        raise SystemExit("the files come from more than one campaign")
    campaign = json.loads(campaigns.pop())
    if (campaign["manifest"], campaign["days"], campaign["limit"]) != \
            (sha256(MANIFEST), days, limit):
        raise SystemExit("the campaign ran another manifest, other days or another limit")
    print(f"campaign: manifest {campaign['manifest'][:12]}, code {campaign['code'][:12]}, "
          f"days {days}, limit {limit}")
    return blocks, times


def write_csv(rows: list, path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Block:
    """One block's rows; values are per resampling unit: a day on map 0, a map on the unseen
    maps (its days averaged)."""

    def __init__(self, rows: list, by_map: bool):
        self.by_map = by_map
        self.cells = defaultdict(dict)  # (method, load, key) -> {unit: [values]}
        for r in rows:
            unit = r["map"] if by_map else r["seed"]
            for key in ("on_time_rate", "served_rate", "expiry_rate") + OPS:
                self.cells[r["method"], r["load"], key].setdefault(unit, []).append(r[key])
        self.rows = rows

    def unit(self, method: str, load: str, key="on_time_rate") -> dict:
        """{unit: value} of one entrant, or of a group averaged over its seeds per day."""
        runs = GROUPS.get(method, [method])
        per_run = [self.cells[run, load, key] for run in runs]
        days = sorted(per_run[0])
        return {u: float(np.mean([np.mean(p[u]) for p in per_run])) for u in days}

    def mean(self, method, load, key="on_time_rate") -> float:
        return float(np.mean(list(self.unit(method, load, key).values())))

    def diff(self, a, b, load, key="on_time_rate") -> tuple:
        ua, ub = self.unit(a, load, key), self.unit(b, load, key)
        return paired_ci(np.array([ua[u] - ub[u] for u in sorted(ua)]))


def reading(c) -> str:
    direction = "better" if c[1] > 0 else "worse" if c[2] < 0 else "no direction"
    return direction + (", practically equivalent" if -EQUIVALENCE < c[1] and c[2] < EQUIVALENCE
                        else "")


def fmt(c) -> str:
    return f"{c[0] * 100:+6.2f} [{c[1] * 100:+6.2f}, {c[2] * 100:+6.2f}]"


def report_block(name: str, b: Block, loads: list, times: dict, out: dict) -> None:
    methods = list(dict.fromkeys(r["method"] for r in b.rows))
    unit = "maps" if b.by_map else "days"
    for load in loads:
        res = out.setdefault(name, {}).setdefault(load, {})
        n = len(b.unit("slack", load))
        print(f"\n== {name}, {load} load, {n} {unit}: mean daily rate (on-time / served / expired)")
        res["means"] = {}
        for m in methods + list(GROUPS):
            if not b.cells[GROUPS.get(m, [m])[0], load, "on_time_rate"]:  # horizon at low load
                continue
            v = [b.mean(m, load, k) for k in ("on_time_rate", "served_rate", "expiry_rate")]
            res["means"][m] = v
            print(f"  {m:24s} {v[0]:.3f} / {v[1]:.3f} / {v[2]:.3f}")
        print("  PPO groups, per-seed means (SD, range in points):")
        res["seeds"] = {}
        for g, runs in GROUPS.items():
            means = [b.mean(r, load) for r in runs]
            res["seeds"][g] = means
            print(f"    {g:16s} " + " ".join(f"{v:.3f}" for v in means)
                  + f"  (SD {np.std(means, ddof=1) * 100:.2f}, range "
                  f"{(max(means) - min(means)) * 100:.2f})")
        print(f"  primary, final agent minus (points, 95 % bootstrap over {unit}):")
        res["primary"] = {}
        for base in PRIMARY:
            if not b.cells[base, load, "on_time_rate"]:
                continue
            c = b.diff("final", base, load)
            if base == "slack":
                target = (f"target >= +{MEDIUM_GAIN * 100:.0f} with lower bound > 0"
                          if load == "medium" else f"margin {SLACK_MARGIN[load] * 100:+.0f}")
                met = (c[0] >= MEDIUM_GAIN and c[1] > 0) if load == "medium" \
                    else c[1] > SLACK_MARGIN[load]
            else:
                target, met = f"margin {PLANNER_MARGIN * 100:+.0f}", c[1] > PLANNER_MARGIN
            res["primary"][base] = [*c, met]
            served = b.diff("final", base, load, "served_rate")
            print(f"    {base:12s} {fmt(c)}  {reading(c)}; {target}: "
                  f"{'met' if met else 'NOT met'}; served {fmt(served)}")
            print(f"    shipped - {base:12s} {fmt(b.diff(SHIPPED, base, load))}")
        print("  secondary, final agent minus:")
        res["secondary"] = {}
        for base in OTHERS:
            c = b.diff("final", base, load)
            res["secondary"][base] = list(c)
            print(f"    {base:12s} {fmt(c)}  {reading(c)}; served "
                  f"{fmt(b.diff('final', base, load, 'served_rate'))}")
        print("  development path and controls:")
        res["contrasts"] = {}
        for label, a, c_b, key in CONTRASTS:
            c = b.diff(a, c_b, load, key)
            res["contrasts"][label] = list(c)
            matched = ""
            if a in GROUPS and c_b in GROUPS and len(GROUPS[c_b]) == 3:
                per_seed = [b.mean(ra, load, key) - b.mean(rb, load, key)
                            for ra, rb in zip(GROUPS[a], GROUPS[c_b])]
                matched = "  seeds " + " ".join(f"{d * 100:+.2f}" for d in per_seed)
            print(f"    {label:38s} {fmt(c)}  {reading(c)}{matched}")
        print("  operations: mean lateness of late deliveries (min, pooled), energy (battery %), "
              "km, empty km, backlog (orders waiting) per day:")
        res["operations"] = {}
        for m in OPS_METHODS:
            runs = GROUPS.get(m, [m])
            mine = [r for r in b.rows if r["method"] in runs and r["load"] == load]
            if not mine:
                continue
            late = [r["delivered"] - r["on_time"] for r in mine]
            lateness = sum(r["mean_lateness"] * n for r, n in zip(mine, late)) / max(1, sum(late))
            ops = [lateness] + [b.mean(m, load, k) for k in OPS]
            res["operations"][m] = ops
            print(f"    {m:12s} lateness {ops[0]:5.1f}  energy {ops[1]:6.1f}  km {ops[2]:5.1f}  "
                  f"empty km {ops[3]:5.1f}  backlog {ops[4]:5.2f}")
        if b.by_map:
            print("  per map, final agent minus (points): min / median / max, maps won of "
                  f"{n}:")
            for base in PRIMARY + ("insertion", "rollout"):
                if not b.cells[base, load, "on_time_rate"]:
                    continue
                ua, ub = b.unit("final", load), b.unit(base, load)
                d = np.array([ua[u] - ub[u] for u in sorted(ua)]) * 100
                res.setdefault("per_map", {})[base] = d.tolist()
                print(f"    {base:12s} {d.min():+6.2f} / {np.median(d):+6.2f} / {d.max():+6.2f}"
                      f"   won {int((d > 0).sum())}")
        print("  pooled on-time by type (on time / released):")
        for m in ["final", "slack", "insertion", "rollout", "horizon"]:
            runs = GROUPS.get(m, [m])
            mine = [r for r in b.rows if r["method"] in runs and r["load"] == load]
            if not mine:
                continue
            pooled = [sum(r[f"on_time_count_{t.name}"] for r in mine)
                      / max(1, sum(r[f"released_{t.name}"] for r in mine)) for t in ORDER_TYPES]
            print(f"    {m:12s} " + "  ".join(f"{t.name} {v:.3f}"
                                              for t, v in zip(ORDER_TYPES, pooled)))
    print(f"\n== {name}, decision times over all decisions, ms (throughput run, parallel "
          "workers; not a latency benchmark): mean / p95")
    for m in methods:
        cells = [f"{load} {np.mean(t):8.2f} / {np.percentile(t, 95):8.2f}"
                 for load in loads if (t := times[name, m, load])]
        print(f"  {m:24s} " + "   ".join(cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out")
    parser.add_argument("--days", choices=["select", "test"], default="test")
    parser.add_argument("--limit", type=int, default=None, help="as given to the run")
    args = parser.parse_args()
    out = Path(args.out)
    blocks, times = load_rows(out, args.days, args.limit)
    write_csv(blocks["map0"], out / "map0.csv")
    write_csv(blocks["maps"], out / "maps.csv")
    summary = {}
    b0, b1 = Block(blocks["map0"], by_map=False), Block(blocks["maps"], by_map=True)
    report_block("map0", b0, list(LOADS), times, summary)
    report_block("maps", b1, list(LOADS), times, summary)
    print("\n== transfer gap: unseen maps minus map 0, on-time points (different days: "
          "descriptive)")
    summary["transfer"] = {}
    rules = [m for m in dict.fromkeys(r["method"] for r in blocks["map0"]) if m in RULES]
    for m in list(GROUPS) + rules:
        gaps = {load: b1.mean(m, load) - b0.mean(m, load) for load in LOADS
                if b0.cells[GROUPS.get(m, [m])[0], load, "on_time_rate"]}
        summary["transfer"][m] = gaps
        print(f"  {m:16s} " + "  ".join(f"{k} {v * 100:+.2f}" for k, v in gaps.items()))
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
