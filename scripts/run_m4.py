"""Run independent frozen M4 learner seeds in local CPU processes.

Each learner keeps one torch thread; process concurrency is explicitly recorded.
This is independent-seed execution, not distributed or shared-policy training.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from medical_uav.training.io import write_json
from run_m3 import check_frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("configs/m4_protocol.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    protocol = json.loads(args.protocol.read_text())
    check_frozen(repo, protocol)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("use a fresh output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    manifest = {"protocol": protocol, "source_commit": commit,
                "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
                "status": "running", "completed_seeds": [], "running_seeds": []}
    write_json(args.output_dir / "manifest.json", manifest)
    started = time.perf_counter()

    def run_seed(seed, config_name):
        check_frozen(repo, protocol)
        if hashlib.sha256((repo / config_name).read_bytes()).hexdigest() != protocol["configuration_sha256"][config_name]:
            raise RuntimeError("frozen configuration changed")
        config = json.loads((repo / config_name).read_text())
        if (config["architecture"] != "structured" or config["learner_seed"] != seed
                or config["total_steps"] != 1024000 or config["eval_interval_updates"] != 100
                or config["threads"] != 1):
            raise RuntimeError("seed configuration violates the frozen protocol")
        print(f"Starting scratch learner seed {seed}, 1,024,000 transitions", flush=True)
        run_dir = (args.output_dir / f"seed_{seed}").resolve()
        with (args.output_dir / f"seed_{seed}.log").open("w") as log:
            subprocess.run(["/usr/bin/time", "-p", sys.executable, str(repo / "scripts/train.py"),
                            "--config", str(repo / config_name), "--output-dir", str(run_dir)],
                           cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True)
        summary = json.loads((run_dir / "summary.json").read_text())
        if summary["transitions"] != 1024000 or summary["updates"] != 1000 or summary["status"] != "complete":
            raise RuntimeError("formal run did not complete the fixed budget")
        print(f"Completed seed {seed}: {summary['wall_seconds']:.1f}s; Dev best={summary['best_score'][0]:.6f}", flush=True)
        return seed

    futures = {}
    try:
        with ThreadPoolExecutor(max_workers=protocol["execution"]["concurrent_learners"]) as pool:
            for seed, config_name in zip(protocol["learner_seeds"], protocol["configurations"]):
                futures[pool.submit(run_seed, seed, config_name)] = seed
                manifest["running_seeds"].append(seed)
            write_json(args.output_dir / "manifest.json", manifest)
            for future in as_completed(futures):
                seed = future.result()
                manifest["completed_seeds"].append(seed)
                manifest["completed_seeds"].sort()
                manifest["running_seeds"].remove(seed)
                manifest["wall_seconds"] = time.perf_counter() - started
                write_json(args.output_dir / "manifest.json", manifest)
    except Exception as error:
        manifest.update(status="failed", error=repr(error), wall_seconds=time.perf_counter() - started)
        write_json(args.output_dir / "manifest.json", manifest)
        raise
    manifest.update(status="complete", running_seeds=[], wall_seconds=time.perf_counter() - started)
    write_json(args.output_dir / "manifest.json", manifest)
    print(f"All five seeds complete; elapsed {manifest['wall_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
