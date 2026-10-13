"""Observation: a global block, N_SLOTS pending-order slots, N_ONBOARD cargo slots (415 values)."""
import numpy as np

from medical_uav.env.actions import N_SLOTS
from medical_uav.env.config import (EPS, FACILITIES, FULL_BATTERY, N_NODES, N_ONBOARD, ORDER_TYPES,
                                    day_length)
from medical_uav.env.orders import Order, demand_multiplier
from medical_uav.env.simulator import Simulator

N_TYPES = len(ORDER_TYPES)
N_FACILITIES = len(FACILITIES)
GLOBAL_DIM = 13 + 2 * N_NODES + N_TYPES  # 31; see _global_features
PENDING_DIM = 12 + N_TYPES + 2 * N_FACILITIES  # 28; see _pending_features
ONBOARD_DIM = 6 + N_TYPES + N_FACILITIES  # 16; see _onboard_features
OBS_DIM = GLOBAL_DIM + N_SLOTS * PENDING_DIM + N_ONBOARD * ONBOARD_DIM  # 415

OBS_BOUND = 10.0  # every feature is clipped to [-OBS_BOUND, OBS_BOUND]

# fixed feature scales (kept constant so a feature means the same thing under any config)
DURATION_SCALE = 30.0  # minutes, for durations and slack
TRAVEL_SCALE = 10.0  # minutes, for travel times to nodes
COUNT_SCALE = 10.0  # orders, for per-type and recent counts
SAME_DEST_SCALE = 5.0  # orders sharing one destination
DEMAND_SCALE = 2.0  # reference demand multiplier (the default peak); values may exceed 1
RECENT_WINDOW = 30.0  # minutes, for "releases in the last 30 min"
FORECAST_STEPS = (30.0, 60.0)  # minutes ahead for the demand profile


def direct_completion_time(sim: Simulator, order: Order) -> float:
    """Optimistic minutes to finish this order alone: travel and handling only.

    Ignores other cargo, expired-cargo disposal and charging, so the real time can be longer.
    """
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
    """Extra minutes the provisional route needs if this order is picked up first."""
    return provisional_route_time(sim, order) - provisional_route_time(sim)


def rank_pending(sim: Simulator, n_slots: int = N_SLOTS) -> list[Order]:
    """Optimistically on-time orders first (by deadline), then the rest (by expiry)."""
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
                for k in range(N_TYPES)]
    recent = sum(1 for r in sim.release_times_log if r > t - RECENT_WINDOW)
    return [
        t / day_length(cfg),  # time of day, 0..1
        max(0.0, cfg.release_window - t) / cfg.release_window,  # share of release window left
        demand_multiplier(t, cfg) / DEMAND_SCALE,  # demand level now
        demand_multiplier(t + FORECAST_STEPS[0], cfg) / DEMAND_SCALE,  # ... in 30 min
        demand_multiplier(t + FORECAST_STEPS[1], cfg) / DEMAND_SCALE,  # ... in 60 min
        sim.lam / cfg.loads["high"],  # today's base load relative to "high"
        sim.battery / FULL_BATTERY,  # battery level
        load / cfg.max_weight,  # cargo weight / capacity
        len(sim.onboard) / cfg.max_boxes,  # boxes on board / capacity
        *_one_hot(sim.location, N_NODES),  # where the drone is (N_NODES values)
        # minutes to each node (N_NODES values)
        *[sim.travel_time(sim.location, n) / TRAVEL_SCALE for n in range(N_NODES)],
        to_charger / FULL_BATTERY,  # battery needed to reach the nearest charger
        len(sim.pending) / cfg.pool_cap,  # pending orders / pool capacity
        (len(sim.pending) - n_visible) / cfg.pool_cap,  # pending orders not shown in the slots
        *per_type,  # pending orders of each type (N_TYPES values)
        recent / COUNT_SCALE,  # orders released in the last 30 min
    ]


def _pending_features(sim: Simulator, o: Order, pick_allowed: bool) -> list[float]:
    t = sim.t
    direct = direct_completion_time(sim, o)
    onboard_dests = {b.dest for b in sim.onboard if b.status == "onboard"}
    same_dest = sum(1 for p in sim.pending if p.dest == o.dest and p is not o)
    return [
        1.0,  # slot is used
        *_one_hot(o.type_idx, N_TYPES),  # order type
        *_one_hot(o.origin, N_FACILITIES),  # pickup facility
        *_one_hot(o.dest, N_FACILITIES),  # destination facility
        o.weight,  # kg
        (o.deadline - t) / o.limit,  # time to deadline, in units of L (negative when late)
        (o.expiry - t) / o.limit,  # time to expiry, in units of L
        o.age(t) / o.limit,  # time since release, in units of L
        direct / DURATION_SCALE,  # optimistic time to serve it alone
        (o.deadline - t - direct) / DURATION_SCALE,  # slack left after serving it alone
        float(o.dest in onboard_dests),  # goes where cargo already goes (batching hint)
        same_dest / SAME_DEST_SCALE,  # other pending orders to the same destination
        float(o.origin == sim.location),  # can be loaded without flying
        insertion_extra_time(sim, o) / DURATION_SCALE,  # extra route time if added
        float(pick_allowed),  # PICK for this slot is allowed by the mask
    ]


def _onboard_features(sim: Simulator, o: Order) -> list[float]:
    t = sim.t
    return [
        1.0,  # slot is used
        *_one_hot(o.type_idx, N_TYPES),  # order type
        *_one_hot(o.dest, N_FACILITIES),  # destination facility
        o.weight,  # kg
        (o.deadline - t) / o.limit,  # time to deadline, in units of L
        (o.expiry - t) / o.limit,  # time to expiry, in units of L
        float(o.status == "expired"),  # expired cargo waiting to be unloaded
        direct_completion_time(sim, o) / DURATION_SCALE,  # optimistic time to deliver it
    ]


def build_observation(sim: Simulator, slots: list[Order], mask: np.ndarray) -> np.ndarray:
    features = _global_features(sim, len(slots))
    for k in range(N_SLOTS):
        if k < len(slots):
            features += _pending_features(sim, slots[k], bool(mask[k]))
        else:
            features += [0.0] * PENDING_DIM
    onboard = sorted(sim.onboard, key=lambda o: (o.deadline, o.id))  # most urgent first
    for k in range(N_ONBOARD):
        if k < len(onboard):
            features += _onboard_features(sim, onboard[k])
        else:
            features += [0.0] * ONBOARD_DIM
    obs = np.clip(np.array(features, dtype=np.float32), -OBS_BOUND, OBS_BOUND)
    assert obs.shape == (OBS_DIM,)
    return obs
