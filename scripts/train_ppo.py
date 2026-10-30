"""Train a policy from scratch with masked PPO on randomly drawn training days at all loads.

Usage:
    python scripts/train_ppo.py --net set --seed 11 --decisions 6e6 --out runs/set_s11

Writes updates.csv (one row per update), evals.csv (greedy on-time rate on selection days)
and a checkpoint at every evaluation.
"""
import argparse
import csv
from dataclasses import replace
import json
import time
from pathlib import Path

import numpy as np
import torch

from medical_uav.agents.networks import NETWORKS
from medical_uav.agents.policy import NetworkPolicy, load_checkpoint, save_checkpoint
from medical_uav.agents.ppo import collect, compute_gae, make_envs, ppo_update, warm_up_value
from medical_uav.env import EnvConfig
from medical_uav.env.config import REWARDS
from medical_uav.evaluation import run_day, run_metadata
from medical_uav.splits import SELECT_MAP_SEEDS, SELECT_SEEDS, TRAIN_MAP_SEEDS

N_ENVS, N_STEPS = 16, 128  # decisions per update = 2048
LR, LR_FLOOR = 3e-4, 0.05  # learning rate, decayed linearly to 5 % of it
ENT, ENT_FLOOR = 0.01, 0.1  # entropy bonus, decayed linearly to 10 % of it
EVAL_LOADS = ("medium", "high")


def evaluate_net(net, days: int, many_maps: bool = False, config=None) -> dict:
    """Mean greedy on-time rate on the first `days` selection days at medium and high load; with
    many_maps the days rotate over the selection maps instead of map 0."""
    policy = NetworkPolicy(net)
    net.eval()
    maps = [SELECT_MAP_SEEDS[k % len(SELECT_MAP_SEEDS)] if many_maps else 0 for k in range(days)]
    config = config or EnvConfig()
    result = {load: float(np.mean([run_day(policy, load, s, replace(config, map_seed=m))["on_time_rate"]
                                   for s, m in zip(list(SELECT_SEEDS)[:days], maps)]))
              for load in EVAL_LOADS}
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
    parser.add_argument("--reward", choices=list(REWARDS), default="v2",
                        help="reward variant (default: v2); v2_expiry* change only the expiry penalty")
    parser.add_argument("--no-travel-table", action="store_true",
                        help="omit only the 7x7 travel-time observation block (452 -> 403)")
    parser.add_argument("--init", default=None, help="start from this checkpoint (e.g. BC)")
    parser.add_argument("--maps", choices=["train"], default=None,
                        help="train: draw each day's map from the training maps (default: map 0)")
    parser.add_argument("--warmup-rollouts", type=int, default=40,
                        help="value warm-up rollouts before PPO when --init is given")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    total, eval_every = int(args.decisions), int(args.eval_every)

    maps = list(TRAIN_MAP_SEEDS) if args.maps == "train" else None
    config = EnvConfig(**REWARDS[args.reward], include_travel_table=not args.no_travel_table)
    envs, state = make_envs(N_ENVS, rng, config, maps)
    obs_dim = envs[0].observation_space.shape[0]
    meta = run_metadata(vars(args))
    if args.init:  # start from a trained network; warm up its critic first
        net, info = load_checkpoint(args.init)
        assert info["kind"] == args.net, f"{args.init} holds a {info['kind']} network"
        if net.obs_dim != obs_dim:
            raise ValueError("Initial checkpoint observation layout does not match environment")
        net.train()
        meta["warmup"] = warm_up_value(net, envs, state, rng, rollouts=args.warmup_rollouts)
        w = meta["warmup"]
        print(f"value warm-up: error against day returns {w['value_error_before']:.3f} -> "
              f"{w['value_error_after']:.3f}, explained variance "
              f"{w['explained_variance_before']:.2f} -> {w['explained_variance_after']:.2f} "
              f"({w['decisions']} decisions, not counted below)", flush=True)
    else:
        net = NETWORKS[args.net](obs_dim=obs_dim)
    (out / "run.json").write_text(json.dumps(meta, indent=1))
    opt = torch.optim.Adam(net.parameters(), lr=LR, eps=1e-5)
    updates = open(out / "updates.csv", "w", newline="")
    evals = open(out / "evals.csv", "w", newline="")
    update_log = csv.writer(updates)
    eval_log = csv.writer(evals)
    update_log.writerow(["decisions", "seconds", "policy_loss", "value_loss", "entropy",
                         "approx_kl", "clip_frac", "train_on_time"])
    eval_log.writerow(["decisions", "seconds"] + [f"on_time_{load}" for load in EVAL_LOADS])

    def evaluate(decisions: int) -> dict:
        result = evaluate_net(net, args.eval_days, many_maps=maps is not None, config=config)
        eval_log.writerow([decisions, round(time.time() - t0, 1)] + list(result.values()))
        evals.flush()
        print(f"{decisions:>9d} decisions  {time.time() - t0:6.0f} s  "
              + "  ".join(f"{k} {v:.3f}" for k, v in result.items()), flush=True)
        return result

    t0, decisions, next_eval = time.time(), 0, eval_every
    if total > 0:
        evaluate(0)  # the untrained network, as the curve's starting point
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
            result = evaluate(decisions)
            save_checkpoint(net, args.net, out / f"ckpt_{decisions:08d}.pt",
                            {"decisions": decisions, "seed": args.seed, "net": args.net,
                             "reward": args.reward, "init": args.init, "maps": args.maps,
                             "obs_dim": obs_dim, "no_travel_table": args.no_travel_table,
                             "git_revision": meta["git_revision"],
                             "eval_days": args.eval_days, **{f"on_time_{k}": v
                                                             for k, v in result.items()}})
            next_eval += eval_every
    updates.close()
    evals.close()


if __name__ == "__main__":
    main()
