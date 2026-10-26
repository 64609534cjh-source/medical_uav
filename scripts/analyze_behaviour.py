"""Compare how policies serve the same days: batching, charging, waiting, peak service.

Writes behaviour.csv (one row per policy, load and day), prints the means, and draws the battery
over one day for each policy and the on-time rate by hour of release.

Methods: slack, rollout, and the energy-study interventions slack100 and prepeak
(medical_uav/baselines/energy.py); --mask-full-charge adds each checkpoint with CHARGE(*, 100)
hidden, labelled <checkpoint>/no100.

Usage:
    python scripts/analyze_behaviour.py --checkpoints runs/m3_set_s11/ckpt_06000640.pt \\
        --methods slack,rollout --out outputs/behaviour
"""
import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure

from medical_uav.agents.policy import NetworkPolicy, load_checkpoint
from medical_uav.baselines.energy import PrePeakTopUp, slack100_policy, without_full_charge
from medical_uav.baselines.heuristics import slack_aware_policy
from medical_uav.baselines.rollout import rollout_policy
from medical_uav.env import MedicalUAVEnv
from medical_uav.env.config import EPS, EnvConfig
from medical_uav.evaluation import checkpoint_labels
from medical_uav.metrics import behaviour_metrics, distance_breakdown, episode_metrics
from medical_uav.splits import SCREEN_SEEDS, SELECT_SEEDS

RULES = {"slack": slack_aware_policy, "rollout": rollout_policy, "slack100": slack100_policy}
SEED_SETS = {"screen": SCREEN_SEEDS, "select": SELECT_SEEDS}
NO_FULL = "#no100"  # suffix of a checkpoint run with full charges hidden
HOURS = 8  # orders are released over 8 hours


def make_policy(name: str):
    """A rule by name, or a checkpoint path as a network policy (path#no100: full charges
    hidden); policy(env, obs, mask). The pre-peak rule is returned as itself too, to count its
    top-ups."""
    if name == "prepeak":
        rule = PrePeakTopUp()
        return (lambda env, obs, mask: rule(env)), rule
    if name in RULES:
        return (lambda env, obs, mask: RULES[name](env)), None
    if name.endswith(NO_FULL):
        return without_full_charge(NetworkPolicy(load_checkpoint(name[:-len(NO_FULL)])[0])), None
    return NetworkPolicy(load_checkpoint(name)[0]), None


def run(job) -> dict:
    """One day of one policy: behaviour metrics, on-time by hour of release, battery trace."""
    torch.set_num_threads(1)
    name, label, load, seed = job
    policy, rule = make_policy(name)
    env = MedicalUAVEnv(load=load)
    obs, info = env.reset(options={"demand_seed": seed})
    done = False
    while not done:
        obs, _, done, _, info = env.step(policy(env, obs, info["action_mask"]))
    sim = env.sim
    released, on_time = [], []  # per hour of release, counted so they can be pooled over days
    for h in range(HOURS):
        mine = [o for o in sim.orders if h * 60 <= o.release < (h + 1) * 60]
        released.append(len(mine))
        on_time.append(sum(1 for o in mine
                           if o.status == "delivered" and o.end_time - o.release <= o.limit + EPS))
    trace = [(s.t0, s.battery0) for s in sim.segments] + [(sim.t, sim.battery)]
    m = episode_metrics(sim)
    return {"method": label, "load": load, "seed": seed, "on_time_rate": m["on_time_rate"],
            "charge_share": m["charge_share"], "km_flown": m["km_flown"],
            **behaviour_metrics(sim, env.decision_log), **distance_breakdown(sim),
            "overrides": len(rule.used) if rule else float("nan"),
            "hourly": (released, on_time), "trace": trace}


def shade_peaks(ax) -> None:
    """Grey bands over the demand peaks, in hours of the day."""
    for k, (a, b) in enumerate(EnvConfig().peak_windows):
        ax.axvspan(8 + a / 60, 8 + b / 60, color="#7a7974", alpha=0.12, lw=0,
                   label="demand peak" if k == 0 else None)


def draw(rows: list[dict], labels: list[str], loads: list[str], out: Path) -> None:
    """Battery over the first day of the last load for each policy; on-time rate by hour of
    release, pooled over days (on-time orders / released orders), for every load. Demand peaks
    are shaded."""
    load = loads[-1]
    day = min(r["seed"] for r in rows)
    fig = Figure(figsize=(8, 3.2), dpi=120)
    ax = fig.add_subplot()
    for label in labels:
        r = next(r for r in rows if r["method"] == label and r["load"] == load and r["seed"] == day)
        t, b = zip(*r["trace"])
        ax.plot(np.array(t) / 60 + 8, b, label=label, lw=1.2)
    shade_peaks(ax)
    ax.set_xlabel("time of day (h)")
    ax.set_ylabel("battery (%)")
    ax.set_title(f"Battery over one day ({load} load, demand seed {day})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "battery_day.png")

    fig = Figure(figsize=(4 * len(loads), 3.2), dpi=120)
    for k, load in enumerate(loads):
        ax = fig.add_subplot(1, len(loads), k + 1)
        for label in labels:
            counts = np.array([r["hourly"] for r in rows
                               if r["method"] == label and r["load"] == load])  # [days, 2, hours]
            released, on_time = counts[:, 0].sum(axis=0), counts[:, 1].sum(axis=0)
            ax.plot(np.arange(HOURS) + 8.5, on_time / np.maximum(released, 1), marker="o",
                    label=label)
        shade_peaks(ax)
        ax.set_xlabel("hour of release")
        ax.set_ylabel("on-time rate (pooled)")
        ax.set_title(f"{load} load")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "on_time_by_hour.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoints", nargs="*", default=[])
    parser.add_argument("--methods", default="slack,rollout")
    parser.add_argument("--mask-full-charge", action="store_true",
                        help="also run every checkpoint with CHARGE(*, 100) hidden")
    parser.add_argument("--loads", default="medium,high")
    parser.add_argument("--seeds", choices=list(SEED_SETS), default="screen")
    parser.add_argument("--days", type=int, default=None, help="use only the first DAYS seeds")
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--out", default="outputs/behaviour")
    args = parser.parse_args()

    named = [(m, m) for m in args.methods.split(",") if m]
    named += list(zip(args.checkpoints, checkpoint_labels(args.checkpoints)))
    if args.mask_full_charge:
        named += [(path + NO_FULL, label + "/no100") for path, label
                  in zip(args.checkpoints, checkpoint_labels(args.checkpoints))]
    seeds = list(SEED_SETS[args.seeds])[:args.days]
    jobs = [(name, label, load, seed) for load in args.loads.split(",") for name, label in named
            for seed in seeds]
    with Pool(args.processes) as pool:
        rows = pool.map(run, jobs, chunksize=1)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = [k for k in rows[0] if k not in ("hourly", "trace")]
    with open(out / "behaviour.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: r[k] for k in keys} for r in rows)
    labels = [label for _, label in named]
    for load in args.loads.split(","):
        print(f"\n{load} load (means over {len(seeds)} days)")
        for label in labels:
            rs = [r for r in rows if r["method"] == label and r["load"] == load]
            means = {k: np.nanmean([r[k] for r in rs]) for k in keys[3:]}
            print(f"  {label:28s} " + "  ".join(f"{k} {v:.3f}" for k, v in means.items()))
    draw(rows, labels, args.loads.split(","), out)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
