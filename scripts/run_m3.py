"""Run the frozen five-seed M3 protocol sequentially, without changing training code."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from medical_uav.training.io import write_json


def check_frozen(repo, protocol):
    for relative, expected in protocol["frozen_file_sha256"].items():
        actual = hashlib.sha256((repo / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"frozen source/dependency file changed: {relative}")
    status = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True)
    if status.stdout.strip():
        raise RuntimeError("commit source/configuration changes before starting the formal runs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("configs/m3_protocol.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    protocol = json.loads(args.protocol.read_text())
    check_frozen(repo, protocol)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("use a fresh output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    manifest = {"protocol": protocol, "source_commit": commit,
                "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
                "status": "running", "completed_seeds": [], "current_seed": None}
    write_json(args.output_dir / "manifest.json", manifest)
    started = time.perf_counter()
    try:
        for seed, config_name in zip(protocol["learner_seeds"], protocol["configurations"]):
            check_frozen(repo, protocol)
            config = json.loads((repo / config_name).read_text())
            if (config["learner_seed"] != seed or config["total_steps"] != 1024000
                    or config["eval_interval_updates"] != 100):
                raise RuntimeError("seed configuration violates the frozen budget/schedule")
            manifest["current_seed"] = seed
            write_json(args.output_dir / "manifest.json", manifest)
            print(f"Starting scratch learner seed {seed}, 1,024,000 transitions", flush=True)
            run_dir = (args.output_dir / f"seed_{seed}").resolve()
            with (args.output_dir / f"seed_{seed}.log").open("w") as log:
                subprocess.run([sys.executable, str(repo / "scripts/train.py"), "--config", str(repo / config_name),
                                "--output-dir", str(run_dir)], cwd=repo, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            summary = json.loads((run_dir / "summary.json").read_text())
            if summary["transitions"] != 1024000 or summary["updates"] != 1000 or summary["status"] != "complete":
                raise RuntimeError("formal run did not complete the fixed budget")
            manifest["completed_seeds"].append(seed)
            manifest["wall_seconds"] = time.perf_counter() - started
            write_json(args.output_dir / "manifest.json", manifest)
            print(f"Completed seed {seed}: {summary['wall_seconds']:.1f}s; Dev best={summary['best_score'][0]:.6f}", flush=True)
    except Exception as error:
        manifest.update({"status": "failed", "error": repr(error), "wall_seconds": time.perf_counter() - started})
        write_json(args.output_dir / "manifest.json", manifest)
        raise
    manifest.update({"status": "complete", "current_seed": None, "wall_seconds": time.perf_counter() - started})
    write_json(args.output_dir / "manifest.json", manifest)
    print(f"All five seeds complete; total wall time {manifest['wall_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
