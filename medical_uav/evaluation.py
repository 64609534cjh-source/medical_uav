"""Run policies on the same days and compare them. A policy is called as
policy(env, obs, mask) and returns a valid action index."""
import platform
from dataclasses import replace
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from medical_uav.env import MedicalUAVEnv
from medical_uav.env.config import ENV_VERSION, EPS, ORDER_TYPES, EnvConfig
from medical_uav.env.observation import OBS_DIM, observation_dim
from medical_uav.metrics import behaviour_metrics, distance_breakdown, episode_metrics


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
    """Name each checkpoint run/file (e.g. m3_set_s11/ckpt_06000640); a submitted model in
    checkpoints/ by its file name alone (e.g. ppo_set)."""
    return [Path(p).stem if Path(p).parent.name == "checkpoints" else
            f"{Path(p).parent.name}/{Path(p).stem}" for p in paths]


def heuristic(fn):
    """Adapt a rule-based policy fn(env) to the evaluation signature."""
    return lambda env, obs, mask: fn(env)


def run_day(policy, load: str, demand_seed: int, config=None, details: bool = False) -> dict:
    """Run one day with the policy and return its metrics, total reward and decision count;
    `config` (an EnvConfig) changes the settings, e.g. the map; `details` adds the behaviour
    metrics, the distance breakdown, the share of decisions with more waiting orders than slots
    and every decision's time."""
    # A checkpoint's layout is intrinsic to the network, including legacy full-table files.
    if hasattr(policy, "net"):
        dim = policy.net.obs_dim
        if dim not in (OBS_DIM, observation_dim(False)):
            raise ValueError(f"Unsupported policy observation dimension: {dim}")
        config = replace(config or EnvConfig(), include_travel_table=dim == OBS_DIM)
    env = MedicalUAVEnv(config=config, load=load)
    obs, info = env.reset(options={"demand_seed": demand_seed})
    total, decisions, done, seconds, overflow = 0.0, 0, False, [], 0
    while not done:
        overflow += len(env.sim.pending) > len(env.slots)  # orders the slots cannot show
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
        row[f"served_count_{order_type.name}"] = sum(o.status == "delivered" for o in mine)
        row[f"expired_count_{order_type.name}"] = sum(o.status == "expired" for o in mine)
    if details:
        row.update(**behaviour_metrics(env.sim, env.decision_log), **distance_breakdown(env.sim))
        row["overflow_share"] = overflow / decisions
        row["decision_ms"] = [1000 * s for s in seconds]
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


def first_sustained_crossing(decisions, scores, target: float):
    """Decisions at the first evaluation reaching target whose next evaluation also reaches it;
    None if there is none (a crossing at the last evaluation cannot be confirmed)."""
    for i in range(len(scores) - 1):
        if scores[i] >= target and scores[i + 1] >= target:
            return decisions[i]
    return None


def curve_mean(decisions, scores, start: float, end: float) -> float:
    """Mean of a learning curve between two logged evaluations: trapezoid area / width."""
    d, s = np.asarray(decisions, float), np.asarray(scores, float)
    keep = (d >= start) & (d <= end)
    d, s = d[keep], s[keep]
    return float(((s[1:] + s[:-1]) / 2 * np.diff(d)).sum() / (d[-1] - d[0]))


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
        values = [v for load in r.values() for v in load.values()]
        if not all(np.isfinite(values)):
            reasons.append("a rate is not finite")
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
