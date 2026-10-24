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
from medical_uav.env.config import ORDER_TYPES
from medical_uav.evaluation import heuristic, run_day, run_metadata, select_checkpoint
from medical_uav.splits import SELECT_SEEDS

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


def evaluate_job(job) -> tuple[str, str, dict]:
    """Evaluate one policy (a checkpoint path, or "slack") at one load on the selection days."""
    torch.set_num_threads(1)
    name, load, seeds = job
    if name == "slack":
        policy = heuristic(slack_aware_policy)
    else:
        policy = NetworkPolicy(load_checkpoint(name)[0])
    return name, load, summarize([run_day(policy, load, seed) for seed in seeds])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="run directories holding ckpt_*.pt files")
    parser.add_argument("--processes", type=int, default=16)
    parser.add_argument("--days", type=int, default=len(SELECT_SEEDS), help="selection days used")
    parser.add_argument("--out", default="runs/selection.json")
    args = parser.parse_args()

    checkpoints = {run: sorted(str(p) for p in Path(run).glob("ckpt_*.pt")) for run in args.runs}
    names = ["slack"] + [c for cs in checkpoints.values() for c in cs]
    with Pool(args.processes) as pool:
        seeds = list(SELECT_SEEDS)[:args.days]
        done = pool.map(evaluate_job, [(n, load, seeds) for n in names for load in LOADS],
                        chunksize=1)
    results = {}
    for name, load, summary in done:
        results.setdefault(name, {})[load] = summary
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
    report = {"metadata": run_metadata({"days": "SELECT_SEEDS", "n_days": args.days}),
              "slack": slack, "runs": selection, "all_results": results}
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"Slack-aware: medium {slack['medium']['on_time_rate']:.3f}  "
          f"high {slack['high']['on_time_rate']:.3f}\nSaved: {args.out}")


if __name__ == "__main__":
    main()
