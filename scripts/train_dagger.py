"""Distil the visible horizon on medium student rollouts with fixed low/high self-replay."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from medical_uav.agents.bc import train_imitation
from medical_uav.agents.dagger import aggregate_datasets, collect_student_data, day_schedule
from medical_uav.agents.policy import NetworkPolicy, load_checkpoint, save_checkpoint
from medical_uav.env import EnvConfig
from medical_uav.env.observation import OBS_DIM
from medical_uav.evaluation import run_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--days-per-iteration", type=int, default=16)
    parser.add_argument("--processes", type=int, default=24)
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke", action="store_true", help="isolated tiny plumbing run")
    args = parser.parse_args()
    if not 1 <= args.processes <= 24:
        parser.error("--processes must be in 1..24")
    medium, replay = day_schedule(args.seed, args.iterations, args.days_per_iteration, args.smoke)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Output directory must be empty; no implicit restart")
    net, info = load_checkpoint(args.init)
    if net.obs_dim != OBS_DIM:
        raise ValueError("Registered DAgger uses the full travel-table observation")
    config = EnvConfig(release_window=20) if args.smoke else EnvConfig()
    meta = run_metadata({**vars(args), "medium_days": medium, "replay_days": replay,
                         "init_sha256": hashlib.sha256(Path(args.init).read_bytes()).hexdigest(),
                         "teacher": "visible_horizon", "teacher_reward": "v2",
                         "foresight": False, "obs_dim": net.obs_dim,
                         "epochs": 1 if args.smoke else 5,
                         "lr": 1e-4, "batch_size": 256})
    meta["torch"] = str(meta["torch"])
    (out / "run.json").write_text(json.dumps(meta, indent=1))
    save_checkpoint(net, info["kind"], out / "ckpt_00000000.pt", {**meta, "iteration": 0,
                                                                "decisions": 0})
    start = time.monotonic()
    data = collect_student_data(NetworkPolicy(net), replay, loads=("low", "high"),
                                config=config, label_with_teacher=False)
    history = []
    for iteration, days in enumerate(medium, 1):
        before = time.monotonic()
        current = collect_student_data(NetworkPolicy(net), days, processes=args.processes,
                                       config=config)
        data = aggregate_datasets(data, current)
        np.savez_compressed(out / f"dataset_{iteration:02d}.npz", **data)
        labelled_seconds = time.monotonic() - before
        losses = train_imitation(net, data, epochs=meta["epochs"], seed=args.seed + iteration,
                                 lr=meta["lr"])
        record = {"iteration": iteration, "decisions": len(data["action"]),
                  "teacher_queries": len(current["label"]), "losses": losses,
                  "collection_seconds": labelled_seconds, "seconds": time.monotonic() - start}
        history.append(record)
        save_checkpoint(net, info["kind"], out / f"ckpt_{iteration:08d}.pt", {**meta, **record})
        (out / "history.json").write_text(json.dumps(history, indent=1, allow_nan=False))
        print(f"iteration {iteration}: {record['decisions']} aggregate decisions, "
              f"{record['teacher_queries']} queries, {record['seconds']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
