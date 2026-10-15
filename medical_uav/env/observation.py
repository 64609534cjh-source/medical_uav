"""Observation vector: global features, then pending-order slots, then cargo slots."""
import numpy as np

from medical_uav.env.actions import N_SLOTS
from medical_uav.env.config import (EPS, FACILITIES, FULL_BATTERY, N_NODES, N_ONBOARD, ORDER_TYPES,
                                    day_length)
from medical_uav.env.orders import Order, demand_multiplier
from medical_uav.env.simulator import Simulator

N_TYPES = len(ORDER_TYPES)
N_FACILITIES = len(FACILITIES)
GLOBAL_DIM = 13 + 2 * N_NODES + N_TYPES  # see _global_features
PENDING_DIM = 12 + N_TYPES + 2 * N_FACILITIES  # see _pending_features
ONBOARD_DIM = 6 + N_TYPES + N_FACILITIES  # see _onboard_features
OBS_DIM = GLOBAL_DIM + N_SLOTS * PENDING_DIM + N_ONBOARD * ONBOARD_DIM

OBS_BOUND = 10.0  # every feature is clipped to [-OBS_BOUND, OBS_BOUND]

# Normalisation scales, fixed so each feature keeps its meaning when EnvConfig changes.
DURATION_SCALE = 30.0  # minutes; scales durations
TRAVEL_SCALE = 10.0  # minutes; scales travel times
COUNT_SCALE = 10.0  # scales order counts
SAME_DEST_SCALE = 5.0  # scales the count of orders sharing a destination
DEMAND_SCALE = 2.0  # scales the demand multiplier
RECENT_WINDOW = 30.0  # minutes; window for counting recent arrivals
FORECAST_STEPS = (30.0, 60.0)  # minutes ahead at which demand is reported


def direct_completion_time(sim: Simulator, order: Order) -> float:
    """Return the minutes to finish this order alone, counting only flight and handling."""
    h = sim.cfg.handling_min
    if order.status == "pending":
        return (sim.travel_time(sim.location, order.origin) + h
                + sim.travel_time(order.origin, order.dest) + h)
    return sim.travel_time(sim.location, order.dest) + h


def provisional_route_time(sim: Simulator, extra: Order | None = None) -> float:
    """Estimate the minutes to deliver all cargo, plus `extra` if given, in deadline order."""
    h = sim.cfg.handling_min
    boxes = [o for o in sim.onboard if o.status == "onboard"]
    here, total = sim.location, 0.0
    if extra is not None:
        total += sim.travel_time(here, extra.origin) + h
        here = extra.origin
        boxes.append(extra)
    boxes.sort(key=lambda o: (o.deadline, o.id))
    visited = []  # destinations in visiting order, without repeats
    for o in boxes:
        if o.dest not in visited:
            visited.append(o.dest)
    for node in visited:
        n_boxes = sum(1 for o in boxes if o.dest == node)
        total += sim.travel_time(here, node) + h * n_boxes
        here = node
    return total


def insertion_extra_time(sim: Simulator, order: Order) -> float:
    """Return the minutes that picking up this order first adds to the delivery route."""
    return provisional_route_time(sim, order) - provisional_route_time(sim)


def rank_pending(sim: Simulator) -> list[Order]:
    """Return the pending orders shown in the slots, in slot order.

    Orders that can meet their deadline by direct_completion_time come first, by deadline;
    the rest follow by expiry time.
    """
    feasible, late = [], []
    for o in sim.pending:
        if sim.t + direct_completion_time(sim, o) <= o.deadline + EPS:
            feasible.append(o)
        else:
            late.append(o)
    feasible.sort(key=lambda o: (o.deadline, o.release, o.id))
    late.sort(key=lambda o: (o.expiry, o.release, o.id))
    return (feasible + late)[:N_SLOTS]


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
        t / day_length(cfg),  # fraction of the day elapsed
        max(0.0, cfg.release_window - t) / cfg.release_window,  # share of release window left
        # demand multiplier now and at each forecast step
        demand_multiplier(t, cfg) / DEMAND_SCALE,
        demand_multiplier(t + FORECAST_STEPS[0], cfg) / DEMAND_SCALE,
        demand_multiplier(t + FORECAST_STEPS[1], cfg) / DEMAND_SCALE,
        sim.lam / cfg.loads["high"],  # today's base rate relative to the high load
        sim.battery / FULL_BATTERY,  # battery level
        load / cfg.max_weight,  # share of weight capacity used
        len(sim.onboard) / cfg.max_boxes,  # share of box capacity used
        *_one_hot(sim.location, N_NODES),  # current node
        # travel time to each node
        *[sim.travel_time(sim.location, n) / TRAVEL_SCALE for n in range(N_NODES)],
        to_charger / FULL_BATTERY,  # battery needed to reach the nearest charger
        len(sim.pending) / cfg.pool_cap,  # share of the pending pool used
        (len(sim.pending) - n_visible) / cfg.pool_cap,  # pending orders not shown in the slots
        *per_type,  # pending orders per type
        recent / COUNT_SCALE,  # orders that arrived within RECENT_WINDOW
    ]


def _pending_features(sim: Simulator, o: Order, pick_allowed: bool) -> list[float]:
    t = sim.t
    direct = direct_completion_time(sim, o)
    onboard_dests = {b.dest for b in sim.onboard if b.status == "onboard"}
    same_dest = sum(1 for p in sim.pending if p.dest == o.dest and p is not o)
    return [
        1.0,  # slot holds an order
        *_one_hot(o.type_idx, N_TYPES),  # order type
        *_one_hot(o.origin, N_FACILITIES),  # pickup facility
        *_one_hot(o.dest, N_FACILITIES),  # destination facility
        o.weight,  # kg
        # the next three are in units of the order's on-time limit
        (o.deadline - t) / o.limit,  # time left to the deadline, negative when late
        (o.expiry - t) / o.limit,  # time left to expiry
        o.age(t) / o.limit,  # time since release
        direct / DURATION_SCALE,  # time to serve this order alone
        (o.deadline - t - direct) / DURATION_SCALE,  # time to spare at the deadline if served alone
        float(o.dest in onboard_dests),  # a box on board has the same destination
        same_dest / SAME_DEST_SCALE,  # other pending orders to the same destination
        float(o.origin == sim.location),  # order waits at the drone's node
        insertion_extra_time(sim, o) / DURATION_SCALE,  # added route time if picked up first
        float(pick_allowed),  # PICK for this slot is valid
    ]


def _onboard_features(sim: Simulator, o: Order) -> list[float]:
    t = sim.t
    return [
        1.0,  # slot holds a box
        *_one_hot(o.type_idx, N_TYPES),  # order type
        *_one_hot(o.dest, N_FACILITIES),  # destination facility
        o.weight,  # kg
        # the next two are in units of the order's on-time limit
        (o.deadline - t) / o.limit,  # time left to the deadline
        (o.expiry - t) / o.limit,  # time left to expiry
        float(o.status == "expired"),  # box has expired
        direct_completion_time(sim, o) / DURATION_SCALE,  # time to deliver this box directly
    ]


def build_observation(sim: Simulator, slots: list[Order], mask: np.ndarray) -> np.ndarray:
    """Return the observation vector for the current state."""
    features = _global_features(sim, len(slots))
    for k, o in enumerate(slots):
        features += _pending_features(sim, o, bool(mask[k]))
    features += [0.0] * (N_SLOTS - len(slots)) * PENDING_DIM  # empty slots
    onboard = sorted(sim.onboard, key=lambda o: (o.deadline, o.id))  # earliest deadline first
    for o in onboard:
        features += _onboard_features(sim, o)
    features += [0.0] * (N_ONBOARD - len(onboard)) * ONBOARD_DIM  # empty slots
    return np.clip(np.array(features, dtype=np.float32), -OBS_BOUND, OBS_BOUND)
