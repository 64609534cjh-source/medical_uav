"""Choose one checkpoint per training run by the rule fixed before training (roadmap section 4).

Every checkpoint of every run is evaluated greedily on the 100 selection days at all loads,
Slack-aware once on the same days; the result and every checkpoint's eligibility are saved.

Usage:
    python scripts/select_checkpoints.py runs/m3_set_s11 runs/m3_set_s22 --out runs/selection.json
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from medical_uav.agents.policy import NetworkPolicy, load_checkpoint
from medical_uav.baselines.heuristics import slack_aware_policy
from medical_uav.env import EnvConfig
from medical_uav.env.config import ORDER_TYPES
from medical_uav.evaluation import heuristic, run_day, run_metadata, select_checkpoint
from medical_uav.splits import SELECT_MAP_SEEDS, SELECT_SEEDS

LOADS = ("low", "medium", "high")


def summarize(rows: list[dict]) -> dict:
    """Mean daily on-time rate and, per order type, the on-time rate pooled over the days."""
    out = {"on_time_rate": float(np.mean([r["on_time_rate"] for r in rows]))}
    for order_type in ORDER_TYPES:
        name = order_type.name
        released = sum(r[f"released_{name}"] for r in rows)
        on_time = sum(r[f"on_time_count_{name}"] for r in rows)
        out[f"on_time_{name}"] = on_time / released if released else float("nan")
    return out


def evaluate_job(job) -> tuple[str, str, list[dict]]:
    """Run one policy (a checkpoint path, or "slack") at one load on the selection days of one
    map; return its rows."""
    torch.set_num_threads(1)
    name, load, map_seed, seeds = job
    if name == "slack":
        policy = heuristic(slack_aware_policy)
    else:
        policy = NetworkPolicy(load_checkpoint(name)[0])
    return name, load, [run_day(policy, load, seed, EnvConfig(map_seed=map_seed)) for seed in seeds]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="run directories holding ckpt_*.pt files")
    parser.add_argument("--processes", type=int, default=16)
    parser.add_argument("--days", type=int, default=len(SELECT_SEEDS),
                        help="selection days used (on each map)")
    parser.add_argument("--maps", choices=["select"], default=None,
                        help="select: the days on each selection map instead of map 0")
    parser.add_argument("--out", default="runs/selection.json")
    args = parser.parse_args()

    checkpoints = {run: sorted(str(p) for p in Path(run).glob("ckpt_*.pt")) for run in args.runs}
    names = ["slack"] + [c for cs in checkpoints.values() for c in cs]
    maps = list(SELECT_MAP_SEEDS) if args.maps == "select" else [0]
    seeds = list(SELECT_SEEDS)[:args.days]
    with Pool(args.processes) as pool:
        done = pool.map(evaluate_job, [(n, load, m, seeds) for n in names for load in LOADS
                                       for m in maps], chunksize=1)
    rows = {}
    for name, load, day_rows in done:
        rows.setdefault((name, load), []).extend(day_rows)
    results = {}
    for (name, load), day_rows in rows.items():
        results.setdefault(name, {})[load] = summarize(day_rows)
    slack = results.pop("slack")

    selection = {}
    for run, paths in checkpoints.items():
        chosen, rows = select_checkpoint({p: results[p] for p in paths}, slack)
        selection[run] = {"selected": chosen, "checkpoints": rows,
                          "selected_results": results.get(chosen)}
        if chosen:
            r = results[chosen]
            print(f"{run}: {Path(chosen).name}  medium {r['medium']['on_time_rate']:.3f}  "
                  f"high {r['high']['on_time_rate']:.3f}")
        else:
            print(f"{run}: no eligible checkpoint")
    report = {"metadata": run_metadata({"days": "SELECT_SEEDS", "n_days": args.days,
                                        "maps": args.maps}),
              "slack": slack, "runs": selection, "all_results": results}
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"Slack-aware: medium {slack['medium']['on_time_rate']:.3f}  "
          f"high {slack['high']['on_time_rate']:.3f}\nSaved: {args.out}")


if __name__ == "__main__":
    main()
