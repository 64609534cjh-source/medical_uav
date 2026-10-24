"""Run policies on the same days and compare them. A policy is called as
policy(env, obs, mask) and returns a valid action index."""
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from medical_uav.env import MedicalUAVEnv
from medical_uav.env.config import ENV_VERSION, EPS, ORDER_TYPES
from medical_uav.metrics import episode_metrics


def run_metadata(extra: dict) -> dict:
    """What produced a result: environment version, code revision, library versions, and extra."""
    try:
        revision = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                  text=True, cwd=Path(__file__).parent).stdout.strip() or "unknown"
    except OSError:
        revision = "unknown"
    return {"env_version": ENV_VERSION, "git_revision": revision,
            "python": platform.python_version(), "torch": torch.__version__, **extra}


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
    total, decisions, done, seconds = 0.0, 0, False, []
    while not done:
        t0 = time.perf_counter()
        a = policy(env, obs, info["action_mask"])
        seconds.append(time.perf_counter() - t0)
        assert info["action_mask"][a], f"invalid action {a}"  # the env would silently WAIT
        obs, reward, done, _, info = env.step(a)
        total += reward
        decisions += 1
    m = episode_metrics(env.sim)
    row = {"load": load, "seed": demand_seed, "env_version": ENV_VERSION, "return": total,
           "decisions": decisions, "decision_ms_mean": 1000 * float(np.mean(seconds)),
           "decision_ms_p95": 1000 * float(np.percentile(seconds, 95))}
    for key, value in m.items():
        if key == "on_time_by_type":
            row.update({f"on_time_{name}": rate for name, rate in value.items()})
        else:
            row[key] = value
    # counts per order type, so rates can be pooled over days
    released = [o for o in env.sim.orders if o.status != "future"]
    for k, order_type in enumerate(ORDER_TYPES):
        mine = [o for o in released if o.type_idx == k]
        row[f"released_{order_type.name}"] = len(mine)
        row[f"on_time_count_{order_type.name}"] = sum(
            1 for o in mine if o.status == "delivered" and o.end_time - o.release <= o.limit + EPS)
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


LOW_LOAD_FLOOR = 0.97  # a selected checkpoint keeps the low load almost fully on time
TYPE_TOLERANCE = 0.03  # and no order type more than 3 points below Slack-aware (medium, high)


def select_checkpoint(results: dict, slack: dict) -> tuple[str | None, list[dict]]:
    """Choose one checkpoint of a run by the rule fixed before training (roadmap section 4).

    `results` maps each checkpoint path to {load: {"on_time_rate": ..., "on_time_<type>": pooled
    rate}} on the selection days, `slack` holds the same for Slack-aware. A checkpoint is eligible
    if its low-load on-time rate is at least LOW_LOAD_FLOOR and, at medium and high load, no order
    type is more than TYPE_TOLERANCE below Slack-aware. The eligible checkpoint with the highest
    mean of the medium and high on-time rates wins; ties go to the earlier checkpoint.
    Returns the chosen path (None if none is eligible) and one row per checkpoint."""
    rows = []
    for path in sorted(results):
        r = results[path]
        reasons = []
        if r["low"]["on_time_rate"] < LOW_LOAD_FLOOR:
            reasons.append(f"low load {r['low']['on_time_rate']:.3f} < {LOW_LOAD_FLOOR}")
        for load in ("medium", "high"):
            for order_type in ORDER_TYPES:
                key = f"on_time_{order_type.name}"
                if r[load][key] < slack[load][key] - TYPE_TOLERANCE:
                    reasons.append(f"{order_type.name} at {load} load {r[load][key]:.3f} vs "
                                   f"Slack-aware {slack[load][key]:.3f}")
        score = (r["medium"]["on_time_rate"] + r["high"]["on_time_rate"]) / 2
        rows.append({"path": path, "eligible": not reasons, "reason": "; ".join(reasons),
                     "score": score})
    eligible = [row for row in rows if row["eligible"]]
    best = max(eligible, key=lambda row: row["score"], default=None)  # first maximum = earliest
    return (best["path"] if best else None), rows
