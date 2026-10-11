"""Event-driven simulation of one operating day (§4.4, §4.5).

The simulator has no notion of action indices or observations; the Gym
wrapper translates actions into the do_* commands below.
"""
import math
from dataclasses import dataclass

from medical_uav.env.config import CHARGERS, EPS, FACILITIES, M, EnvConfig
from medical_uav.env.map_gen import CityMap
from medical_uav.env.orders import Order
from medical_uav.env.physics import charge_minutes, delivery_reward, flight_energy, flight_time


@dataclass
class Segment:
    """One piece of the drone's timeline, recorded for metrics and rendering."""

    kind: str  # fly / load / unload / dispose / charge / wait
    t0: float
    t1: float
    node_from: int
    node_to: int
    battery0: float
    battery1: float
    order_id: int | None = None


class Simulator:
    def __init__(self, cfg: EnvConfig, city_map: CityMap, orders: list[Order], lam: float):
        self.cfg = cfg
        self.city_map = city_map
        self.orders = orders
        self.lam = lam

        self.t = 0.0
        self.location = M
        self.battery = 100.0
        self.onboard: list[Order] = []  # includes expired cargo until it is disposed
        self.pending: list[Order] = []

        self.segments: list[Segment] = []
        self.events: list[tuple[float, str, int]] = []  # (time, kind, order id)
        self.release_times_log: list[float] = []
        self.total_reward = 0.0
        self.energy_used = 0.0
        self.km_flown = 0.0
        self.charging_time = 0.0

        self._next_release = 0  # index of the next order to be released
        self.advance_to(0.0)

    # ------------------------------------------------------------------ queries

    def load_kg(self) -> float:
        return sum(o.weight for o in self.onboard)

    def energy(self, a: int, b: int, load: float) -> float:
        return flight_energy(self.city_map.dist_km[a, b], load, self.cfg)

    def travel_time(self, a: int, b: int) -> float:
        return flight_time(self.city_map.dist_km[a, b], self.cfg)

    def nearest_charger(self, node: int, load: float) -> tuple[int, float]:
        """(charger, energy to reach it) with the lowest energy; ties go to M."""
        best, best_energy = None, math.inf
        for c in CHARGERS:
            e = self.energy(node, c, load)
            if e < best_energy - EPS:
                best, best_energy = c, e
        return best, best_energy

    def next_event_time(self) -> float:
        """Next order release or expiry, inf if nothing is left to happen."""
        t_next = math.inf
        if self._next_release < len(self.orders):
            t_next = self.orders[self._next_release].release
        for o in self.pending:
            t_next = min(t_next, o.expiry)
        for o in self.onboard:
            if o.status == "onboard":
                t_next = min(t_next, o.expiry)
        return t_next

    def is_done(self) -> bool:
        if self.t < self.cfg.release_window - EPS:
            return False
        if self._next_release < len(self.orders) or self.pending:
            return False
        return all(o.status != "onboard" for o in self.onboard)

    def check_invariant(self) -> None:
        """Battery must always cover the reserve plus the flight to the nearest charger."""
        _, to_charger = self.nearest_charger(self.location, self.load_kg())
        need = self.cfg.reserve + to_charger
        assert self.battery >= need - 1e-6, f"battery {self.battery:.3f} < {need:.3f} at t={self.t:.2f}"

    # ------------------------------------------------------------------ events

    def advance_to(self, t_target: float, strict: bool = False) -> None:
        """Process releases and expiries up to t_target, then move the clock there.

        strict=True leaves events exactly at t_target unprocessed (used so that an
        unload finishing at t_target is recorded before an expiry at t_target).
        At equal times expiries are processed before releases.
        """
        while True:
            t_release = math.inf
            if self._next_release < len(self.orders):
                t_release = self.orders[self._next_release].release
            t_expiry = min([o.expiry for o in self.pending]
                           + [o.expiry for o in self.onboard if o.status == "onboard"],
                           default=math.inf)
            t_next = min(t_release, t_expiry)
            if strict and t_next >= t_target - EPS:
                break
            if not strict and t_next > t_target + EPS:
                break

            if t_expiry <= t_release + EPS:
                self._expire_orders(t_expiry)
            else:
                self._release_next_order()

        self.t = max(self.t, t_target)

    def _expire_orders(self, t: float) -> None:
        for o in list(self.pending):
            if o.expiry <= t + EPS:
                self.pending.remove(o)
                self._mark_expired(o)
        for o in self.onboard:
            if o.status == "onboard" and o.expiry <= t + EPS:
                self._mark_expired(o)  # stays on board as expired cargo

    def _mark_expired(self, o: Order) -> None:
        o.status = "expired"
        o.end_time = o.expiry
        self.total_reward -= self.cfg.expiry_penalty
        self.events.append((o.expiry, "expire", o.id))

    def _release_next_order(self) -> None:
        o = self.orders[self._next_release]
        self._next_release += 1
        self.release_times_log.append(o.release)
        if len(self.pending) >= self.cfg.pool_cap:
            o.status = "rejected"
            o.end_time = o.release
            self.total_reward -= self.cfg.expiry_penalty
            self.events.append((o.release, "reject", o.id))
        else:
            o.status = "pending"
            self.pending.append(o)
            self.events.append((o.release, "release", o.id))

    # ------------------------------------------------------------------ movement

    def fly_to(self, node: int) -> None:
        """Fly to node; at a facility, expired cargo is unloaded on arrival (§4.4)."""
        if node != self.location:
            e = self.energy(self.location, node, self.load_kg())
            t_arrive = self.t + self.travel_time(self.location, node)
            self.segments.append(Segment("fly", self.t, t_arrive, self.location, node,
                                         self.battery, self.battery - e))
            self.battery -= e
            self.energy_used += e
            self.km_flown += self.city_map.dist_km[self.location, node]
            self.total_reward -= self.cfg.energy_cost * e
            self.advance_to(t_arrive)
            self.location = node
        if node in FACILITIES:
            self.dispose_expired_cargo()

    def dispose_expired_cargo(self) -> None:
        for o in [o for o in self.onboard if o.status == "expired"]:
            self._handle_box("dispose", o)
            o.disposed_time = self.t
            self.onboard.remove(o)
            self.events.append((self.t, "dispose", o.id))

    def _handle_box(self, kind: str, o: Order, strict: bool = False) -> None:
        """Spend handling_min on the ground for one box (load/unload/dispose)."""
        t_end = self.t + self.cfg.handling_min
        self.segments.append(Segment(kind, self.t, t_end, self.location, self.location,
                                     self.battery, self.battery, o.id))
        self.advance_to(t_end, strict=strict)

    # ------------------------------------------------------------------ commands

    def do_pick(self, order: Order) -> None:
        self.fly_to(order.origin)
        assert order.status == "pending", f"order {order.id} is {order.status}"
        self._handle_box("load", order)
        assert order.status == "pending", f"order {order.id} expired while loading"
        order.status = "onboard"
        order.pickup_time = self.t
        self.pending.remove(order)
        self.onboard.append(order)
        self.events.append((self.t, "pickup", order.id))
        self.check_invariant()

    def do_deliver(self, node: int) -> None:
        """Unload every box for node, most urgent first; each box is rewarded when unloaded."""
        self.fly_to(node)
        boxes = sorted([o for o in self.onboard if o.dest == node], key=lambda o: (o.deadline, o.id))
        for o in boxes:
            if o.status == "expired":
                self._handle_box("dispose", o)
                o.disposed_time = self.t
                self.events.append((self.t, "dispose", o.id))
            else:
                # strict: an unload ending exactly at expiry counts as a delivery (§4.4)
                self._handle_box("unload", o, strict=True)
                if o.status == "expired":  # expired during the unload
                    o.disposed_time = self.t
                    self.events.append((self.t, "dispose", o.id))
                else:
                    o.status = "delivered"
                    o.end_time = self.t
                    self.total_reward += delivery_reward(o.age(self.t), o.limit, o.expiry_after,
                                                         self.cfg)
                    self.events.append((self.t, "deliver", o.id))
            self.onboard.remove(o)
        self.advance_to(self.t)  # events at this exact time
        self.check_invariant()

    def do_charge(self, station: int, target: float) -> None:
        self.fly_to(station)
        t_end = self.t + charge_minutes(self.battery, target, self.cfg)
        self.segments.append(Segment("charge", self.t, t_end, station, station,
                                     self.battery, target))
        self.charging_time += t_end - self.t
        self.advance_to(t_end)
        self.battery = target
        self.check_invariant()

    def do_return(self) -> None:
        self.fly_to(M)
        self.check_invariant()

    def do_wait(self) -> None:
        """Wait until the next release/expiry, at most wait_max minutes."""
        t_end = min(self.next_event_time(), self.t + self.cfg.wait_max)
        self.segments.append(Segment("wait", self.t, t_end, self.location, self.location,
                                     self.battery, self.battery))
        self.advance_to(t_end)
        self.check_invariant()
