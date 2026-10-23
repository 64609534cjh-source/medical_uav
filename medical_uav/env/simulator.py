"""Event-driven simulation of one day; each do_* method carries out one action.
Times are minutes from the start of the day; battery is in %."""
import math
from dataclasses import dataclass

from medical_uav.env.config import CHARGERS, EPS, FACILITIES, FULL_BATTERY, M, EnvConfig
from medical_uav.env.map_gen import CityMap
from medical_uav.env.orders import Order
from medical_uav.env.physics import charge_minutes, delivery_reward, flight_energy, flight_time

INVARIANT_TOL = 1e-6  # rounding tolerance for check_invariant, battery %


@dataclass
class Segment:
    """One interval of the drone's activity, recorded for metrics and rendering."""

    kind: str  # one of fly, load, unload, dispose, charge, wait
    t0: float
    t1: float
    node_from: int
    node_to: int
    battery0: float
    battery1: float
    order_id: int | None = None


class Simulator:
    """State of the drone and all orders, advanced one action at a time."""

    def __init__(self, cfg: EnvConfig, city_map: CityMap, orders: list[Order], lam: float):
        self.cfg = cfg
        self.city_map = city_map
        self.orders = orders
        self.lam = lam  # base arrival rate, orders per minute

        self.t = 0.0  # minutes since the start of the day
        self.location = M
        self.battery = FULL_BATTERY
        self.onboard: list[Order] = []  # includes expired boxes not yet unloaded
        self.pending: list[Order] = []

        self.segments: list[Segment] = []
        self.events: list[tuple[float, str, int]] = []  # (time, kind, order id)
        self.release_times_log: list[float] = []  # arrival times
        self.total_reward = 0.0
        self.energy_used = 0.0  # total battery % used for flying
        self.km_flown = 0.0
        self.charging_time = 0.0

        self._next_release = 0  # index of the next order to be released
        self.advance_to(0.0)

    def load_kg(self) -> float:
        """Return the total weight on board."""
        return sum(o.weight for o in self.onboard)

    def energy(self, a: int, b: int, load: float) -> float:
        """Return the battery % used to fly from a to b carrying load kg."""
        return flight_energy(self.city_map.dist_km[a, b], load, self.cfg)

    def travel_time(self, a: int, b: int) -> float:
        """Return the minutes needed to fly from a to b."""
        return flight_time(self.city_map.dist_km[a, b], self.cfg)

    def nearest_charger(self, node: int, load: float) -> tuple[int, float]:
        """Return the charger cheapest to reach from node and the battery % needed; ties go to M."""
        best, best_energy = None, math.inf
        for c in CHARGERS:
            e = self.energy(node, c, load)
            if e < best_energy - EPS:
                best, best_energy = c, e
        return best, best_energy

    def next_event_time(self) -> float:
        """Return the time of the next release or expiry, or inf if none is left."""
        return min(self._next_release_time(), self._next_expiry_time())

    def _next_release_time(self) -> float:
        if self._next_release < len(self.orders):
            return self.orders[self._next_release].release
        return math.inf

    def _next_expiry_time(self) -> float:
        times = [o.expiry for o in self.pending]
        times += [o.expiry for o in self.onboard if o.status == "onboard"]
        return min(times, default=math.inf)

    def is_done(self) -> bool:
        """Return True once no more orders can arrive and every order is closed."""
        if self.t < self.cfg.release_window - EPS:
            return False
        if self._next_release < len(self.orders) or self.pending:
            return False
        return all(o.status != "onboard" for o in self.onboard)

    def check_invariant(self) -> None:
        """Assert that the battery covers the reserve plus the flight to the nearest charger."""
        _, to_charger = self.nearest_charger(self.location, self.load_kg())
        need = self.cfg.reserve + to_charger
        assert self.battery >= need - INVARIANT_TOL, \
            f"battery {self.battery:.3f} < {need:.3f} at t={self.t:.2f}"

    def advance_to(self, t_target: float) -> None:
        """Process releases and expiries up to and including t_target, then move the clock there.
        An expiry and a release at the same time are processed expiry first."""
        while True:
            t_release = self._next_release_time()
            t_expiry = self._next_expiry_time()
            if min(t_release, t_expiry) > t_target + EPS:
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
        o.status = "pending"
        self.pending.append(o)
        self.events.append((o.release, "release", o.id))

    def fly_to(self, node: int) -> None:
        """Fly to node; at a facility, expired cargo is unloaded on arrival."""
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
        """Unload all expired boxes at the current node."""
        for o in [o for o in self.onboard if o.status == "expired"]:
            self._handle_box("dispose", o)
            self._dispose(o)

    def _dispose(self, o: Order) -> None:
        """Remove expired box o from the drone; its handling time is already counted."""
        o.disposed_time = self.t
        self.onboard.remove(o)
        self.events.append((self.t, "dispose", o.id))

    def _handle_box(self, kind: str, o: Order) -> None:
        """Spend the handling time for one box on the ground."""
        t_end = self.t + self.cfg.handling_min
        self.segments.append(Segment(kind, self.t, t_end, self.location, self.location,
                                     self.battery, self.battery, o.id))
        self.advance_to(t_end)

    def do_pick(self, order: Order) -> None:
        """Fly to the order's origin and load it."""
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
        """Fly to node and unload every box for it, earliest deadline first."""
        self.fly_to(node)
        boxes = [o for o in self.onboard if o.dest == node]
        boxes.sort(key=lambda o: (o.deadline, o.id))
        for o in boxes:
            if o.status == "expired":
                self._handle_box("dispose", o)
                self._dispose(o)
                continue
            self._handle_box("unload", o)
            if o.status == "expired":  # expired during the unload
                self._dispose(o)
                continue
            o.status = "delivered"
            o.end_time = self.t
            self.total_reward += delivery_reward(o.age(self.t), o.limit, self.cfg)
            self.onboard.remove(o)
            self.events.append((self.t, "deliver", o.id))
        self.advance_to(self.t)  # process events due now
        self.check_invariant()

    def do_charge(self, station: int, target: float) -> None:
        """Fly to station and charge up to target battery %."""
        self.fly_to(station)
        t_end = self.t + charge_minutes(self.battery, target, self.cfg)
        self.segments.append(Segment("charge", self.t, t_end, station, station,
                                     self.battery, target))
        self.charging_time += t_end - self.t
        self.advance_to(t_end)
        self.battery = target
        self.check_invariant()

    def do_return(self) -> None:
        """Fly back to the medical center."""
        self.fly_to(M)
        self.check_invariant()

    def do_wait(self) -> None:
        """Wait until the next release or expiry, for at most wait_max minutes."""
        t_end = min(self.next_event_time(), self.t + self.cfg.wait_max)
        self.segments.append(Segment("wait", self.t, t_end, self.location, self.location,
                                     self.battery, self.battery))
        self.advance_to(t_end)
        self.check_invariant()
