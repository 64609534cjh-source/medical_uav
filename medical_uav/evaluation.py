"""Run policies on the same days and compare them. A policy is called as
policy(env, obs, mask) and returns a valid action index."""
from pathlib import Path

import numpy as np

from medical_uav.env import MedicalUAVEnv
from medical_uav.env.config import ENV_VERSION
from medical_uav.metrics import episode_metrics


def checkpoint_labels(paths: list) -> list[str]:
    """Name each checkpoint by its file name, or by folder/file name when file names repeat."""
    stems = [Path(p).stem for p in paths]
    return [f"{Path(p).parent.name}/{s}" if stems.count(s) > 1 else s for p, s in zip(paths, stems)]


def heuristic(fn):
    """Adapt a rule-based policy fn(env) to the evaluation signature."""
    return lambda env, obs, mask: fn(env)


def run_day(policy, load: str, demand_seed: int) -> dict:
    """Run one day with the policy and return its metrics, total reward and decision count."""
    env = MedicalUAVEnv(load=load)
    obs, info = env.reset(options={"demand_seed": demand_seed})
    total, decisions, done = 0.0, 0, False
    while not done:
        a = policy(env, obs, info["action_mask"])
        assert info["action_mask"][a], f"invalid action {a}"  # the env would silently WAIT
        obs, reward, done, _, info = env.step(a)
        total += reward
        decisions += 1
    m = episode_metrics(env.sim)
    row = {"load": load, "seed": demand_seed, "env_version": ENV_VERSION, "return": total,
           "decisions": decisions}
    for key, value in m.items():
        if key == "on_time_by_type":
            row.update({f"on_time_{name}": rate for name, rate in value.items()})
        else:
            row[key] = value
    return row


def evaluate(policies: dict, loads: list[str], seeds) -> list[dict]:
    """Run every policy on every (load, demand seed) day; one row per method and day."""
    rows = []
    for load in loads:
        for seed in seeds:
            for name, policy in policies.items():
                rows.append({"method": name, **run_day(policy, load, seed)})
    return rows


def paired_ci(diffs: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of per-day differences and the 2.5 % / 97.5 % points of its bootstrap over days."""
    diffs = np.asarray(diffs, dtype=float)
    boot = np.random.default_rng(seed).choice(diffs, (n_boot, len(diffs))).mean(axis=1)
    return float(diffs.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
