"""Behaviour cloning of Slack-aware: record 900 teacher days, train both networks, save them.

Usage:
    python scripts/train_bc.py            # writes checkpoints/bc_mlp.pt, bc_set.pt, bc_history.json
"""
import argparse
import json
import time
from pathlib import Path

import torch

from medical_uav.agents.bc import collect_demonstrations, split_by_seed, train_bc
from medical_uav.agents.networks import NETWORKS
from medical_uav.agents.policy import save_checkpoint
from medical_uav.baselines.heuristics import slack_aware_policy

REPO = Path(__file__).resolve().parents[1]
LOADS = ["low", "medium", "high"]
DEMAND_SEEDS = range(300)  # training demand seeds; 300 seeds x 3 loads = 900 teacher days


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--processes", type=int, default=16, help="workers for recording days")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=str(REPO / "checkpoints"))
    args = parser.parse_args()

    t0 = time.time()
    data = collect_demonstrations(slack_aware_policy, LOADS, DEMAND_SEEDS, args.processes)
    train, val = split_by_seed(data, val_fraction=0.2, seed=0)
    print(f"{len(data['action'])} decisions from {len(DEMAND_SEEDS) * len(LOADS)} days "
          f"({len(train['action'])} train, {len(val['action'])} validation), "
          f"{time.time() - t0:.0f} s")

    histories = {}
    for kind, net_class in NETWORKS.items():
        t0 = time.time()
        net = net_class()
        history = train_bc(net, train, val, epochs=args.epochs, device=args.device)
        best = min(range(len(history)), key=lambda e: history[e]["val_loss"])
        info = {"teacher": "slack-aware", "loads": LOADS,
                "demand_seeds": [0, len(DEMAND_SEEDS) - 1],
                "decisions": len(data["action"]), "val_fraction": 0.2, "epochs": args.epochs,
                "best_epoch": best + 1, "val_acc": history[best]["val_acc"],
                "val_acc_by_family": history[best]["val_acc_by_family"]}
        save_checkpoint(net, kind, Path(args.out) / f"bc_{kind}.pt", info)
        histories[kind] = history
        print(f"{kind}: best epoch {best + 1}, validation accuracy {history[best]['val_acc']:.3f}, "
              f"{time.time() - t0:.0f} s")
    with open(Path(args.out) / "bc_history.json", "w") as f:
        json.dump(histories, f, indent=1)


if __name__ == "__main__":
    torch.set_num_threads(8)
    main()
