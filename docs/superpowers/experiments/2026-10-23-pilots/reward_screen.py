"""Spike: one-step rollout over Slack-aware, visible orders only, scored by three reward candidates."""
import copy, json, sys
from multiprocessing import Pool
import numpy as np
from medical_uav.env import MedicalUAVEnv
from medical_uav.env.config import EPS
from medical_uav.baselines.heuristics import slack_aware_policy
from medical_uav.metrics import episode_metrics
from medical_uav.splits import DEV_SEEDS


def visible_copy(env):
    memo = {id(env.city_map): env.city_map, id(env.cfg): env.cfg}
    c = copy.deepcopy(env, memo)
    sim = c.sim
    keep = {o.id for o in c.slots} | {o.id for o in sim.onboard}
    sim.pending = [o for o in sim.pending if o.id in keep]
    sim.orders = [o for o in sim.orders[:sim._next_release] if o.id in keep or o.status in ("delivered", "expired")]
    sim._next_release = len(sim.orders)
    return c


def score(sim, kind):
    on_time = late = expired = 0
    for o in sim.orders:
        if o.status == "delivered":
            if o.end_time - o.release <= o.limit + EPS:
                on_time += 1
            else:
                late += 1
        elif o.status == "expired":
            expired += 1
    if kind == "current":
        return sim.total_reward
    if kind == "ontime":
        return on_time - 1e-4 * sim.energy_used  # energy only breaks ties
    return on_time + 0.1 * late - 0.1 * expired - 0.001 * sim.energy_used  # compromise


def rollout_value(env, a, kind):
    c = visible_copy(env)
    c.step(a)
    while c.sim.pending or any(o.status == "onboard" for o in c.sim.onboard):
        c.step(slack_aware_policy(c))
    return score(c.sim, kind)


def rollout_policy(env, kind):
    base = slack_aware_policy(env)
    best_a, best_v = base, rollout_value(env, base, kind)
    for a in np.flatnonzero(env.action_masks()):
        if a != base:
            v = rollout_value(env, int(a), kind)
            if v > best_v + 1e-9:
                best_a, best_v = int(a), v
    return best_a


def run(job):
    load, seed, kind = job
    env = MedicalUAVEnv(load=load)
    env.reset(options={"demand_seed": seed})
    done = False
    while not done:
        a = slack_aware_policy(env) if kind == "slack" else rollout_policy(env, kind)
        _, _, done, _, _ = env.step(a)
    m = episode_metrics(env.sim)
    return dict(load=load, seed=seed, kind=kind, on_time=m["on_time_rate"], served=m["served_rate"],
                expired=m["expiry_rate"], ret=env.sim.total_reward, **{f"type_{k}": v for k, v in m["on_time_by_type"].items()})


if __name__ == "__main__":
    kinds = ["slack", "current", "ontime", "compromise"]
    jobs = [(l, s, k) for l in ("medium", "high") for s in DEV_SEEDS for k in kinds]
    with Pool(18) as pool:
        rows = pool.map(run, jobs, chunksize=1)
    json.dump(rows, open(sys.argv[1], "w"))
    for load in ("medium", "high"):
        base = {r["seed"]: r["on_time"] for r in rows if r["load"] == load and r["kind"] == "slack"}
        for k in kinds:
            rs = [r for r in rows if r["load"] == load and r["kind"] == k]
            d = [r["on_time"] - base[r["seed"]] for r in rs]
            types = " ".join(f"{t[5:]}={np.nanmean([r[t] for r in rs]):.2f}" for t in rs[0] if t.startswith("type_"))
            print(f"{load:6s} {k:10s} on-time {np.mean([r['on_time'] for r in rs]):.3f}  served {np.mean([r['served'] for r in rs]):.3f}  "
                  f"expired {np.mean([r['expired'] for r in rs]):.3f}  diff-vs-slack {np.mean(d):+.3f}  {types}")
