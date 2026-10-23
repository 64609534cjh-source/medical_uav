"""Headroom check: one-step rollout over Slack-aware using only orders already released.

usage: python scripts/headroom.py <loads, e.g. medium,high> <n_dev_days> [reward|ontime|late<w>] [visible]
"""
import copy
import sys
import time
from multiprocessing import Pool

import numpy as np

from medical_uav.env import MedicalUAVEnv
from medical_uav.baselines.heuristics import slack_aware_policy
from medical_uav.metrics import episode_metrics
from medical_uav.splits import DEV_SEEDS

SCORE = sys.argv[3] if len(sys.argv) > 3 else "reward"
VISIBLE_ONLY = len(sys.argv) > 4 and sys.argv[4] == "visible"


def known_copy(env):
    memo = {id(env.city_map): env.city_map, id(env.cfg): env.cfg}
    c = copy.deepcopy(env, memo)
    c.sim.orders = c.sim.orders[:c.sim._next_release]  # forget orders not yet released
    if VISIBLE_ONLY:  # also forget pending orders outside the 12 visible slots
        keep = {o.id for o in c.slots}
        hidden = [o for o in c.sim.pending if o.id not in keep]
        c.sim.pending = [o for o in c.sim.pending if o.id in keep]
        hidden_ids = {o.id for o in hidden}
        c.sim.orders = [o for o in c.sim.orders if o.id not in hidden_ids]
        c.sim._next_release = len(c.sim.orders)
    return c


def rollout_value(env, a):
    c = known_copy(env)
    c.step(a)
    sim = c.sim
    while sim.pending or any(o.status == "onboard" for o in sim.onboard):
        c.step(slack_aware_policy(c))
    if SCORE == "ontime":
        eps = 1e-9
        on_time = sum(1 for o in sim.orders if o.status == "delivered" and o.end_time - o.release <= o.limit + eps)
        return on_time + 1e-3 * sim.total_reward
    if SCORE.startswith("late"):  # on-time +1, late delivery +w, expired or rejected -0.5
        w = float(SCORE[4:])
        eps = 1e-9
        on_time = late = lost = 0
        for o in sim.orders:
            if o.status == "delivered":
                if o.end_time - o.release <= o.limit + eps:
                    on_time += 1
                else:
                    late += 1
            elif o.status in ("expired", "rejected"):
                lost += 1
        return on_time + w * late - 0.5 * lost
    return sim.total_reward


def rollout_policy(env):
    base = slack_aware_policy(env)
    best_a, best_v = base, rollout_value(env, base)
    for a in np.flatnonzero(env.action_masks()):
        if a == base:
            continue
        v = rollout_value(env, int(a))
        if v > best_v + 1e-9:
            best_a, best_v = int(a), v
    return best_a


def run(args):
    load, seed, name = args
    env = MedicalUAVEnv(load=load)
    env.reset(options={"demand_seed": seed})
    pol = rollout_policy if name == "rollout" else slack_aware_policy
    t0, done, n = time.time(), False, 0
    while not done:
        _, _, done, _, _ = env.step(pol(env))
        n += 1
    m = episode_metrics(env.sim)
    return load, seed, name, m["on_time_rate"], env.sim.total_reward, (time.time() - t0) / n, m["served_rate"], m["expiry_rate"], m["mean_lateness"]


if __name__ == "__main__":
    loads = sys.argv[1].split(",")
    n_days = int(sys.argv[2])
    jobs = [(l, s, p) for l in loads for s in list(DEV_SEEDS)[:n_days] for p in ("slack", "rollout")]
    with Pool(16) as pool:
        res = pool.map(run, jobs)
    import json, os
    if os.environ.get("DUMP"):
        json.dump([[x[0], x[1], x[2], x[3], x[6]] for x in res], open(os.environ["DUMP"], "w"))
    for load in loads:
        for p in ("slack", "rollout"):
            r = [x for x in res if x[0] == load and x[2] == p]
            print(f"{load:6s} {p:8s} on-time {np.mean([x[3] for x in r]):.3f}  return {np.mean([x[4] for x in r]):6.2f}"
                  f"  served {np.mean([x[6] for x in r]):.3f}  expired {np.mean([x[7] for x in r]):.3f}  late-min {np.mean([x[8] for x in r]):.1f}")
        d = [x[3] - y[3] for x in res for y in res if x[0] == y[0] == load and x[1] == y[1] and x[2] == "rollout" and y[2] == "slack"]
        print(f"{load:6s} paired diff rollout - slack: mean {np.mean(d):+.3f}, wins {sum(v > 1e-9 for v in d)}/{len(d)}")
