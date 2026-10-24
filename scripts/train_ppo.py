"""Train a policy from scratch with masked PPO on randomly drawn training days at all loads.

Usage:
    python scripts/train_ppo.py --net set --seed 11 --decisions 6e6 --out runs/set_s11

Writes updates.csv (one row per update), evals.csv (greedy on-time rate on selection days)
and a checkpoint at every evaluation.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from medical_uav.agents.networks import NETWORKS
from medical_uav.agents.policy import NetworkPolicy, save_checkpoint
from medical_uav.agents.ppo import collect, compute_gae, make_envs, ppo_update
from medical_uav.evaluation import run_day, run_metadata
from medical_uav.splits import SELECT_SEEDS

N_ENVS, N_STEPS = 16, 128  # decisions per update = 2048
LR, LR_FLOOR = 3e-4, 0.05  # learning rate, decayed linearly to 5 % of it
ENT, ENT_FLOOR = 0.01, 0.1  # entropy bonus, decayed linearly to 10 % of it
EVAL_LOADS = ("medium", "high")


def evaluate_net(net, days: int) -> dict:
    """Mean greedy on-time rate on the first `days` selection days at medium and high load."""
    policy = NetworkPolicy(net)
    net.eval()
    result = {load: float(np.mean([run_day(policy, load, s)["on_time_rate"]
                                   for s in list(SELECT_SEEDS)[:days]])) for load in EVAL_LOADS}
    net.train()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--net", choices=list(NETWORKS), default="set")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--decisions", type=float, default=6e6)
    parser.add_argument("--eval-every", type=float, default=5e5)
    parser.add_argument("--eval-days", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--reward", default="v2", help="reward variant")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps(run_metadata(vars(args)), indent=1))
    total, eval_every = int(args.decisions), int(args.eval_every)

    net = NETWORKS[args.net]()
    opt = torch.optim.Adam(net.parameters(), lr=LR, eps=1e-5)
    envs, state = make_envs(N_ENVS, rng)
    updates = open(out / "updates.csv", "w", newline="")
    evals = open(out / "evals.csv", "w", newline="")
    update_log = csv.writer(updates)
    eval_log = csv.writer(evals)
    update_log.writerow(["decisions", "seconds", "policy_loss", "value_loss", "entropy",
                         "approx_kl", "clip_frac", "train_on_time"])
    eval_log.writerow(["decisions", "seconds"] + [f"on_time_{load}" for load in EVAL_LOADS])

    t0, decisions, next_eval = time.time(), 0, eval_every
    while decisions < total:
        buf = collect(envs, net, state, N_STEPS, rng)
        decisions += N_ENVS * N_STEPS
        left = 1 - decisions / total  # fraction of training still to come
        for group in opt.param_groups:
            group["lr"] = LR * max(left, LR_FLOOR)
        adv, ret = compute_gae(buf.reward, buf.value, buf.done, buf.last_value)
        stats = ppo_update(net, opt, buf, adv, ret, ent_coef=ENT * max(left, ENT_FLOOR), rng=rng)
        finished, state["finished"] = state["finished"], []  # days completed in this rollout
        train_on_time = np.mean([m["on_time_rate"] for m in finished]) if finished else np.nan
        update_log.writerow([decisions, round(time.time() - t0, 1)]
                            + [round(stats[k], 5) for k in ("policy_loss", "value_loss", "entropy",
                                                            "approx_kl", "clip_frac")]
                            + [round(train_on_time, 4)])
        updates.flush()
        if decisions >= next_eval or decisions >= total:
            result = evaluate_net(net, args.eval_days)
            eval_log.writerow([decisions, round(time.time() - t0, 1)] + list(result.values()))
            evals.flush()
            save_checkpoint(net, args.net, out / f"ckpt_{decisions:08d}.pt",
                            {"decisions": decisions, "seed": args.seed, "net": args.net,
                             "eval_days": args.eval_days, **{f"on_time_{k}": v
                                                             for k, v in result.items()}})
            print(f"{decisions:>9d} decisions  {time.time() - t0:6.0f} s  "
                  + "  ".join(f"{k} {v:.3f}" for k, v in result.items()), flush=True)
            next_eval += eval_every
    updates.close()
    evals.close()


if __name__ == "__main__":
    main()
