"""Order records and generation of a day's orders."""
from dataclasses import dataclass

import numpy as np

from medical_uav.env.config import HOSPITALS, M, ORDER_TYPES, EnvConfig


@dataclass
class Order:
    """One delivery request and its progress."""

    id: int
    type_idx: int  # index into ORDER_TYPES
    origin: int  # pickup node
    dest: int  # delivery node
    weight: float  # kg
    release: float  # time the order appears, minutes
    limit: float  # minutes after release within which delivery is on time
    expiry_after: float  # minutes after release at which the order expires
    status: str = "future"  # future / pending / onboard / delivered / expired / rejected
    pickup_time: float | None = None
    end_time: float | None = None  # time the order was delivered, expired or rejected
    disposed_time: float | None = None  # time expired cargo was unloaded

    @property
    def deadline(self) -> float:  # latest on-time delivery time
        return self.release + self.limit

    @property
    def expiry(self) -> float:  # time the order expires
        return self.release + self.expiry_after

    def age(self, t: float) -> float:
        return t - self.release


def demand_multiplier(t: float, cfg: EnvConfig) -> float:
    """Return the arrival-rate multiplier at time t; zero outside the release window."""
    if t < 0 or t >= cfg.release_window:
        return 0.0
    for start, end in cfg.peak_windows:
        if start <= t < end:
            return cfg.peak_multiplier
    return cfg.offpeak_multiplier


def _segments(cfg: EnvConfig):
    """Split the release window into intervals with a constant multiplier."""
    bounds = {0.0, cfg.release_window}
    for start, end in cfg.peak_windows:
        bounds.update((start, end))
    bounds = sorted(bounds)
    return [(a, b, demand_multiplier(a, cfg)) for a, b in zip(bounds[:-1], bounds[1:])]


def _sample_route(rng, route: str) -> tuple[int, int]:
    """Draw an (origin, destination) pair for a route category."""
    hospitals = list(HOSPITALS)
    if route == "center_to_hospital":
        return M, int(rng.choice(hospitals))
    if route == "hospital_to_center":
        return int(rng.choice(hospitals)), M
    origin = int(rng.choice(hospitals))
    dest = int(rng.choice([h for h in hospitals if h != origin]))
    return origin, dest


def generate_orders(demand_seed: int, lam: float, cfg: EnvConfig) -> list[Order]:
    """Generate one day's orders sorted by release time; lam is the base rate per minute."""
    rng = np.random.default_rng(demand_seed)

    # Poisson arrivals: draw a count for each interval, then spread the times uniformly
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
                            order_type.limit, order_type.expiry_after))
    return orders
