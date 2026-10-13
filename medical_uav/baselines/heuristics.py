"""Rule-based baselines. Policies take the env and return a valid action index."""
import numpy as np

from medical_uav.env.actions import CHARGE0, CHARGE_TARGETS, DELIVER0, PICK0, WAIT, charge_action
from medical_uav.env.config import CHARGERS, EPS, FACILITIES
from medical_uav.env.masks import predict_arrival
from medical_uav.env.observation import direct_completion_time
from medical_uav.metrics import episode_metrics

DEFAULT_THETA = 20.0  # charge when battery < this % (best of 20/30/40 on dev days)
IDLE_CHARGE_BELOW = 80.0  # with no PICK/DELIVER possible, top up when battery < this %
# charge targets tried in order: those at or above the idle threshold (80 %, then 100 %)
CHARGE_PREFERENCE = tuple(t for t in CHARGE_TARGETS if t >= IDLE_CHARGE_BELOW)


def random_policy(env, rng: np.random.Generator) -> int:
    return int(rng.choice(np.flatnonzero(env.action_masks())))


def charging_rule(env, theta: float) -> int | None:
    """Shared charging rule: when wanted, charge at the nearest station (80 %, else 100 %)."""
    sim, mask = env.sim, env.action_masks()
    has_service = mask[PICK0:CHARGE0].any()
    wanted = sim.battery < theta or (not has_service and sim.battery < IDLE_CHARGE_BELOW)
    if not wanted:
        return None
    nearest, _ = sim.nearest_charger(sim.location, sim.load_kg())
    stations = [nearest] + [c for c in CHARGERS if c != nearest]
    for station in stations:
        for target in CHARGE_PREFERENCE:
            a = charge_action(station, target)
            if mask[a]:
                return a
    return None


def edf_policy(env, theta: float = DEFAULT_THETA) -> int:
    """Earliest deadline first over valid pickups and deliveries."""
    a = charging_rule(env, theta)
    if a is not None:
        return a
    sim, mask = env.sim, env.action_masks()
    candidates = []  # (deadline, deliver-before-pick, action)
    for k, order in enumerate(env.slots):
        if mask[PICK0 + k]:
            candidates.append((order.deadline, 1, PICK0 + k))
    for n in FACILITIES:
        if mask[DELIVER0 + n]:
            deadline = min(o.deadline for o in sim.onboard if o.dest == n and o.status == "onboard")
            candidates.append((deadline, 0, DELIVER0 + n))
    return min(candidates)[2] if candidates else WAIT


def slack_aware_policy(env, theta: float = DEFAULT_THETA) -> int:
    """Deliver here, batch, charge if needed, then EDF that puts hopeless work last."""
    sim, mask = env.sim, env.action_masks()

    # 1) boxes for the current node: unload them now (no flight needed)
    if sim.location in FACILITIES and mask[DELIVER0 + sim.location]:
        return DELIVER0 + sim.location

    # 2) batching: load an order waiting here that goes where cargo already goes
    onboard_dests = {o.dest for o in sim.onboard if o.status == "onboard"}
    for k, order in enumerate(env.slots):
        if mask[PICK0 + k] and order.origin == sim.location and order.dest in onboard_dests:
            return PICK0 + k

    # 3) shared charging rule
    a = charging_rule(env, theta)
    if a is not None:
        return a

    # 4) work that can still be on time first, then earliest deadline
    candidates = []  # (hopeless, deadline, deliver-before-pick, action)
    for k, order in enumerate(env.slots):
        if mask[PICK0 + k]:
            hopeless = sim.t + direct_completion_time(sim, order) > order.deadline + EPS
            candidates.append((hopeless, order.deadline, 1, PICK0 + k))
    for n in FACILITIES:
        if mask[DELIVER0 + n]:
            deadline = min(o.deadline for o in sim.onboard if o.dest == n and o.status == "onboard")
            hopeless = not _delivery_can_be_on_time(sim, n)
            candidates.append((hopeless, deadline, 0, DELIVER0 + n))
    return min(candidates)[3] if candidates else WAIT


def _delivery_can_be_on_time(sim, n: int) -> bool:
    """True if at least one box for n would still be on time when unloaded there."""
    t, expired = predict_arrival(sim, n)
    t += sim.cfg.handling_min * len(expired)  # expired cargo is unloaded first
    boxes = [o for o in sim.onboard if o.dest == n and o not in expired]
    for o in sorted(boxes, key=lambda o: (o.deadline, o.id)):
        t += sim.cfg.handling_min
        if t <= o.deadline + EPS:
            return True
    return False


def run_episode(env, policy, demand_seed: int, options: dict | None = None) -> dict:
    """Run one day with policy(env) -> action; returns metrics plus return and decisions."""
    env.reset(options={**(options or {}), "demand_seed": demand_seed})
    total, decisions, done = 0.0, 0, False
    while not done:
        _, reward, done, _, _ = env.step(policy(env))
        total += reward
        decisions += 1
    result = episode_metrics(env.sim)
    result["return"] = total
    result["decisions"] = decisions
    return result
