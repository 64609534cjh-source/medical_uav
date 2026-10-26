"""Compare 1/2/4 Torch threads in separate processes using full PPO update cycles."""

import argparse
import json
import statistics
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from medical_uav.training.io import write_json
from medical_uav.training.trainer import TrainConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup-updates", type=int, default=5)
    parser.add_argument("--measured-updates", type=int, default=20)
    args = parser.parse_args()
    if args.warmup_updates < 0 or args.measured_updates < 1:
        parser.error("nonnegative warmup and positive measurement window required")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("choose a fresh output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    results = []
    for threads in (1, 2, 4):
        config = TrainConfig(total_steps=(args.warmup_updates + args.measured_updates) * 1024,
                             threads=threads, eval_split="train", evaluate_initial=False,
                             evaluate_baselines=False)
        config_path = args.output_dir / f"threads_{threads}.json"
        write_json(config_path, asdict(config))
        output = args.output_dir / f"threads_{threads}"
        with (args.output_dir / f"threads_{threads}.log").open("w") as log:
            subprocess.run([sys.executable, str(repo / "scripts/train.py"), "--config", str(config_path.resolve()),
                            "--output-dir", str(output.resolve())], cwd=repo, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        rows = [json.loads(line) for line in (output / "updates.jsonl").read_text().splitlines()]
        measured = rows[args.warmup_updates:]
        # Stable window excludes evaluation but includes collection, GAE, optimizer
        # and checkpoint stages. Full wall time is retained separately, including eval.
        cycles = [sum(r[k] for k in ("collect_seconds", "gae_seconds", "update_seconds", "checkpoint_seconds"))
                  for r in measured]
        result = {"threads": threads, "warmup_updates": args.warmup_updates,
                  "measured_updates": len(measured), "transitions_per_update": 1024,
                  "mean_cycle_seconds_excluding_eval": statistics.mean(cycles),
                  "median_cycle_seconds_excluding_eval": statistics.median(cycles),
                  "window_wall_seconds": rows[-1]["wall_seconds"] - (
                      rows[args.warmup_updates - 1]["wall_seconds"] if args.warmup_updates else 0),
                  "stage_mean_seconds": {k: statistics.mean(r[k] for r in measured) for k in (
                      "collect_seconds", "gae_seconds", "update_seconds", "checkpoint_seconds", "eval_seconds")}}
        results.append(result)
        print(f"{threads} threads: {statistics.mean(cycles):.4f} seconds/update", flush=True)
    write_json(args.output_dir / "summary.json", {
        "purpose": "timing comparison on one Medium learner seed; not a learning experiment",
        "results": results,
        "fastest_threads": min(results, key=lambda r: r["mean_cycle_seconds_excluding_eval"])["threads"],
        "scope": "One sequential pass per thread count; timings may vary with other PC workloads.",
    })


if __name__ == "__main__":
    main()
