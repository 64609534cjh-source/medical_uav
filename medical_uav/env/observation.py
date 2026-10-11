"""415-dim observation: global block, 12 pending slots, 3 onboard slots."""
import numpy as np

from medical_uav.env.config import EPS, FACILITIES, N_NODES, ORDER_TYPES, day_length
from medical_uav.env.orders import Order, demand_multiplier
from medical_uav.env.simulator import Simulator

GLOBAL_DIM = 31
PENDING_DIM = 28
ONBOARD_DIM = 16
N_SLOTS = 12
N_ONBOARD = 3
OBS_DIM = GLOBAL_DIM + N_SLOTS * PENDING_DIM + N_ONBOARD * ONBOARD_DIM

OBS_BOUND = 10.0  # features are clipped to [-OBS_BOUND, OBS_BOUND]

# feature scales
DURATION_SCALE = 30.0  # minutes, for durations and slack
TRAVEL_SCALE = 10.0  # minutes, for travel times to nodes
COUNT_SCALE = 10.0  # orders, for per-type and recent counts
SAME_DEST_SCALE = 5.0
RECENT_WINDOW = 30.0  # minutes, for 'releases in the last 30 min'
FORECAST_STEPS = (30.0, 60.0)  # minutes ahead for the demand profile


def direct_completion_time(sim: Simulator, order: Order) -> float:
    """Time to finish this order alone from the current position, handling included."""
    h = sim.cfg.handling_min
    if order.status == "pending":
        return (sim.travel_time(sim.location, order.origin) + h
                + sim.travel_time(order.origin, order.dest) + h)
    return sim.travel_time(sim.location, order.dest) + h


def provisional_route_time(sim: Simulator, extra: Order | None = None) -> float:
    """Visit destinations of onboard boxes (plus extra) in earliest-deadline order."""
    h = sim.cfg.handling_min
    boxes = [o for o in sim.onboard if o.status == "onboard"]
    here, total = sim.location, 0.0
    if extra is not None:
        total += sim.travel_time(here, extra.origin) + h
        here = extra.origin
        boxes.append(extra)
    boxes.sort(key=lambda o: (o.deadline, o.id))
    visited = []
    for o in boxes:
        if o.dest not in visited:
            visited.append(o.dest)
    for node in visited:
        n_boxes = sum(1 for o in boxes if o.dest == node)
        total += sim.travel_time(here, node) + h * n_boxes
        here = node
    return total


def insertion_extra_time(sim: Simulator, order: Order) -> float:
    return provisional_route_time(sim, order) - provisional_route_time(sim)


def rank_pending(sim: Simulator, n_slots: int = N_SLOTS) -> list[Order]:
    """Orders that can still be on time first (by deadline), then late ones (by expiry)."""
    feasible, late = [], []
    for o in sim.pending:
        if sim.t + direct_completion_time(sim, o) <= o.deadline + EPS:
            feasible.append(o)
        else:
            late.append(o)
    feasible.sort(key=lambda o: (o.deadline, o.release, o.id))
    late.sort(key=lambda o: (o.expiry, o.release, o.id))
    return (feasible + late)[:n_slots]


def _one_hot(index: int, size: int) -> list[float]:
    v = [0.0] * size
    v[index] = 1.0
    return v


def _global_features(sim: Simulator, n_visible: int) -> list[float]:
    cfg, t = sim.cfg, sim.t
    load = sim.load_kg()
    _, to_charger = sim.nearest_charger(sim.location, load)
    per_type = [sum(1 for o in sim.pending if o.type_idx == k) / COUNT_SCALE
                for k in range(len(ORDER_TYPES))]
    recent = sum(1 for r in sim.release_times_log if r > t - RECENT_WINDOW)
    return [
        t / day_length(cfg),
        max(0.0, cfg.release_window - t) / cfg.release_window,
        demand_multiplier(t, cfg) / 2.0,
        demand_multiplier(t + FORECAST_STEPS[0], cfg) / 2.0,
        demand_multiplier(t + FORECAST_STEPS[1], cfg) / 2.0,
        sim.lam / cfg.loads["high"],
        sim.battery / 100.0,
        load / cfg.max_weight,
        len(sim.onboard) / cfg.max_boxes,
        *_one_hot(sim.location, N_NODES),
        *[sim.travel_time(sim.location, n) / TRAVEL_SCALE for n in range(N_NODES)],
        to_charger / 100.0,
        len(sim.pending) / cfg.pool_cap,
        (len(sim.pending) - n_visible) / cfg.pool_cap,
        *per_type,
        recent / COUNT_SCALE,
    ]


def _pending_features(sim: Simulator, o: Order, pick_allowed: bool) -> list[float]:
    t = sim.t
    direct = direct_completion_time(sim, o)
    onboard_dests = {b.dest for b in sim.onboard if b.status == "onboard"}
    same_dest = sum(1 for p in sim.pending if p.dest == o.dest and p is not o)
    return [
        1.0,
        *_one_hot(o.type_idx, len(ORDER_TYPES)),
        *_one_hot(o.origin, len(FACILITIES)),
        *_one_hot(o.dest, len(FACILITIES)),
        o.weight,
        (o.deadline - t) / o.limit,
        (o.expiry - t) / o.limit,
        o.age(t) / o.limit,
        direct / DURATION_SCALE,
        (o.deadline - t - direct) / DURATION_SCALE,
        float(o.dest in onboard_dests),
        same_dest / SAME_DEST_SCALE,
        float(o.origin == sim.location),
        insertion_extra_time(sim, o) / DURATION_SCALE,
        float(pick_allowed),
    ]


def _onboard_features(sim: Simulator, o: Order) -> list[float]:
    t = sim.t
    return [
        1.0,
        *_one_hot(o.type_idx, len(ORDER_TYPES)),
        *_one_hot(o.dest, len(FACILITIES)),
        o.weight,
        (o.deadline - t) / o.limit,
        (o.expiry - t) / o.limit,
        float(o.status == "expired"),
        direct_completion_time(sim, o) / DURATION_SCALE,
    ]


def build_observation(sim: Simulator, slots: list[Order], mask: np.ndarray) -> np.ndarray:
    features = _global_features(sim, len(slots))
    for k in range(N_SLOTS):
        if k < len(slots):
            features += _pending_features(sim, slots[k], bool(mask[k]))
        else:
            features += [0.0] * PENDING_DIM
    onboard = sorted(sim.onboard, key=lambda o: (o.deadline, o.id))
    for k in range(N_ONBOARD):
        if k < len(onboard):
            features += _onboard_features(sim, onboard[k])
        else:
            features += [0.0] * ONBOARD_DIM
    obs = np.clip(np.array(features, dtype=np.float32), -OBS_BOUND, OBS_BOUND)
    assert obs.shape == (OBS_DIM,)
    return obs
