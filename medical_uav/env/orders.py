"""Orders and the seeded daily order stream (§4.3)."""
from dataclasses import dataclass

import numpy as np

from medical_uav.env.config import HOSPITALS, M, ORDER_TYPES, EnvConfig


@dataclass
class Order:
    id: int
    type_idx: int
    origin: int
    dest: int
    weight: float
    release: float
    limit: float  # L
    expiry_after: float  # H
    status: str = "future"  # future / pending / onboard / delivered / expired / rejected
    pickup_time: float | None = None
    end_time: float | None = None  # when delivered, expired or rejected
    disposed_time: float | None = None  # expired cargo unloaded

    @property
    def deadline(self) -> float:
        return self.release + self.limit

    @property
    def expiry(self) -> float:
        return self.release + self.expiry_after

    def age(self, t: float) -> float:
        return t - self.release


def demand_multiplier(t: float, cfg: EnvConfig) -> float:
    """Arrival-rate multiplier m(t): peak / off-peak, zero after the release window."""
    if t < 0 or t >= cfg.release_window:
        return 0.0
    for start, end in cfg.peak_windows:
        if start <= t < end:
            return cfg.peak_multiplier
    return cfg.offpeak_multiplier


def _segments(cfg: EnvConfig):
    """Split [0, release_window) into pieces with a constant multiplier."""
    bounds = {0.0, cfg.release_window}
    for start, end in cfg.peak_windows:
        bounds.update((start, end))
    bounds = sorted(bounds)
    return [(a, b, demand_multiplier(a, cfg)) for a, b in zip(bounds[:-1], bounds[1:])]


def _sample_route(rng, route: str) -> tuple[int, int]:
    hospitals = list(HOSPITALS)
    if route == "center_to_hospital":
        return M, int(rng.choice(hospitals))
    if route == "hospital_to_center":
        return int(rng.choice(hospitals)), M
    origin = int(rng.choice(hospitals))
    dest = int(rng.choice([h for h in hospitals if h != origin]))
    return origin, dest


def generate_orders(demand_seed: int, lam: float, cfg: EnvConfig) -> list[Order]:
    """All orders of one day, sorted by release time. lam is the base rate (orders/min)."""
    rng = np.random.default_rng(demand_seed)

    # piecewise-constant Poisson process: count per segment, uniform times inside it
    releases = []
    for start, end, mult in _segments(cfg):
        n = rng.poisson(lam * mult * (end - start))
        releases.extend(rng.uniform(start, end, size=n))
    releases.sort()

    shares = [t.share for t in ORDER_TYPES]
    orders = []
    for i, release in enumerate(releases):
        type_idx = int(rng.choice(len(ORDER_TYPES), p=shares))
        order_type = ORDER_TYPES[type_idx]
        origin, dest = _sample_route(rng, order_type.route)
        weight = float(rng.uniform(*order_type.weight_range))
        orders.append(Order(i, type_idx, origin, dest, weight, float(release),
                            order_type.limit, order_type.expiry))
    return orders
