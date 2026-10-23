"""Action mask: which actions are valid in the current state.

An action is valid only if the drone can still reach a charger without using the reserve.
"""
import numpy as np

from medical_uav.env.actions import CHARGE0, CHARGE_TABLE, DELIVER0, N_ACTIONS, PICK0, RETURN, WAIT
from medical_uav.env.config import EPS, FACILITIES, M
from medical_uav.env.orders import Order
from medical_uav.env.simulator import Simulator


def predict_arrival(sim: Simulator, node: int) -> tuple[float, list[Order]]:
    """Return the arrival time at node and the boxes on board that will have expired by then."""
    t_arrive = sim.t + sim.travel_time(sim.location, node)
    expired = [o for o in sim.onboard if o.status == "expired" or o.expiry <= t_arrive + EPS]
    return t_arrive, expired


def _battery_ok(sim: Simulator, flight_energy: float, end_node: int, end_load: float) -> bool:
    """Return True if, after this flight, a charger is reachable without using the reserve."""
    _, to_charger = sim.nearest_charger(end_node, end_load)
    return sim.battery - flight_energy - to_charger >= sim.cfg.reserve - EPS


def can_pick(sim: Simulator, order: Order) -> bool:
    """Return True if the order fits the cargo limits, can be loaded before it expires,
    and the flight is safe."""
    cfg = sim.cfg
    t_arrive, expired = predict_arrival(sim, order.origin)
    remaining = [o for o in sim.onboard if o not in expired]
    if len(remaining) + 1 > cfg.max_boxes:
        return False
    new_load = sum(o.weight for o in remaining) + order.weight
    if new_load > cfg.max_weight + EPS:
        return False
    # expired boxes are unloaded first, then the new box is loaded
    t_loaded = t_arrive + cfg.handling_min * (len(expired) + 1)
    if t_loaded >= order.expiry - EPS:
        return False
    e = sim.energy(sim.location, order.origin, sim.load_kg())
    return _battery_ok(sim, e, order.origin, new_load)


def can_deliver(sim: Simulator, node: int) -> bool:
    """Return True if a box for node will not have expired on arrival and the flight is safe.
    The box may still expire while it is being unloaded."""
    _, expired = predict_arrival(sim, node)
    remaining = [o for o in sim.onboard if o not in expired]
    if not any(o.dest == node for o in remaining):
        return False
    load_after = sum(o.weight for o in remaining if o.dest != node)
    e = sim.energy(sim.location, node, sim.load_kg())
    return _battery_ok(sim, e, node, load_after)


def can_charge(sim: Simulator, station: int, target: float) -> bool:
    """Return True if the station is reachable and charging would add enough battery."""
    arrival_battery = sim.battery - sim.energy(sim.location, station, sim.load_kg())
    if arrival_battery < sim.cfg.reserve - EPS:
        return False
    return target > arrival_battery + sim.cfg.min_charge_gain + EPS


def can_return(sim: Simulator) -> bool:
    """Return True if the drone is away from the medical center and can fly back safely."""
    if sim.location == M:
        return False
    return sim.battery - sim.energy(sim.location, M, sim.load_kg()) >= sim.cfg.reserve - EPS


def compute_action_mask(sim: Simulator, slots: list[Order]) -> np.ndarray:
    """Return one flag per action: True if the action is allowed now."""
    mask = np.zeros(N_ACTIONS, dtype=bool)
    for k, order in enumerate(slots):
        mask[PICK0 + k] = can_pick(sim, order)
    for n in FACILITIES:
        mask[DELIVER0 + n] = can_deliver(sim, n)
    for i, (station, target) in enumerate(CHARGE_TABLE):
        mask[CHARGE0 + i] = can_charge(sim, station, target)
    mask[RETURN] = can_return(sim)
    mask[WAIT] = True
    return mask
