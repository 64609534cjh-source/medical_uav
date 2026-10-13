"""Environment settings: fixed structure (defines the observation/action layout) and EnvConfig."""
import math
import numbers
from dataclasses import dataclass, field

# Fixed structure

# Node indices; charger C1 sits at the center M.
NODE_NAMES = ("M", "H1", "H2", "H3", "H4", "H5", "C2")
M, H1, H2, H3, H4, H5, C2 = range(len(NODE_NAMES))
N_NODES = len(NODE_NAMES)
FACILITIES = (M, H1, H2, H3, H4, H5)  # can load/unload (C2 cannot)
HOSPITALS = (H1, H2, H3, H4, H5)
CHARGERS = (M, C2)

N_ONBOARD = 3  # cargo slots in the observation (>= max_boxes)
FULL_BATTERY = 100.0  # battery in %
EPS = 1e-9  # float tolerance


@dataclass(frozen=True)
class OrderType:
    name: str
    route: str  # center_to_hospital / hospital_to_center / hospital_to_hospital
    weight_range: tuple[float, float]  # kg
    limit: float  # L: on time if delivered within this many minutes
    expiry_after: float  # H: lost after this many minutes
    share: float  # fraction of orders


ORDER_TYPES = (
    OrderType("plasma", "center_to_hospital", (0.6, 0.9), limit=30.0, expiry_after=60.0,
              share=0.20),
    OrderType("medicine", "center_to_hospital", (0.3, 0.6), limit=90.0, expiry_after=180.0,
              share=0.30),
    OrderType("sample", "hospital_to_center", (0.2, 0.4), limit=60.0, expiry_after=120.0,
              share=0.30),
    OrderType("supplies", "hospital_to_hospital", (0.4, 0.8), limit=75.0, expiry_after=150.0,
              share=0.20),
)


# Tunable settings

@dataclass
class EnvConfig:
    """Tunable settings; defaults are the main experiment. The env keeps its own copy."""

    # map
    grid_size: int = 16  # cells per side
    cell_km: float = 0.25  # km per cell
    n_zones_range: tuple[int, int] = (2, 4)  # no-fly zones (min, max)
    zone_size_range: tuple[int, int] = (2, 5)  # zone side in cells (min, max)
    min_node_spacing_km: float = 0.75
    c2_min_dist_from_m_km: float = 1.0
    map_seed: int = 0  # same seed -> same map
    max_map_tries: int = 100

    # drone
    speed_km_per_min: float = 1.0  # 60 km/h, load-independent
    energy_per_km: float = 3.5  # battery % per km, empty
    payload_factor: float = 0.25  # +25 % energy per kg of cargo
    reserve: float = 15.0  # battery % always kept
    charge_rate: float = 5.0  # battery % per minute
    min_charge_gain: float = 5.0  # min battery % a CHARGE must add
    handling_min: float = 2.0  # minutes per box (load / unload / dispose)
    max_boxes: int = 3
    max_weight: float = 2.0  # kg

    # demand: rate(t) = base rate x peak or off-peak multiplier
    release_window: float = 480.0  # minutes of order release (08:00-16:00)
    # peaks 09:00-11:00 and 13:00-14:30
    peak_windows: tuple[tuple[float, float], ...] = ((60.0, 180.0), (300.0, 390.0))
    peak_multiplier: float = 2.0
    offpeak_multiplier: float = 0.6
    # base rate in orders/min (about 23 / 47 / 70 orders per day)
    loads: dict = field(default_factory=lambda: {"low": 0.04, "medium": 0.08, "high": 0.12})
    pool_cap: int = 30  # max waiting orders; more are rejected

    # actions
    wait_max: float = 10.0  # max WAIT minutes

    # reward
    reward_on_time: float = 1.0  # R, per on-time box
    expiry_penalty: float = 0.5  # P, per expired or rejected order
    energy_cost: float = 0.001  # per battery % flown

    def validate(self) -> None:
        """Raise ValueError for settings the environment cannot run with."""
        for name in ["grid_size", "map_seed", "max_map_tries", "max_boxes", "pool_cap"]:
            _check_int(name, getattr(self, name))
        for name in ["grid_size", "cell_km", "speed_km_per_min", "energy_per_km", "charge_rate",
                     "handling_min", "max_weight", "release_window", "wait_max", "max_map_tries",
                     "pool_cap"]:
            _check_number(name, getattr(self, name), positive=True)
        for name in ["map_seed", "min_node_spacing_km", "c2_min_dist_from_m_km", "payload_factor",
                     "min_charge_gain", "peak_multiplier", "offpeak_multiplier", "expiry_penalty",
                     "energy_cost", "reward_on_time", "reserve"]:
            _check_number(name, getattr(self, name), positive=False)
        if not self.reserve < FULL_BATTERY:
            raise ValueError(f"reserve must be < {FULL_BATTERY}, got {self.reserve}")
        if not 1 <= self.max_boxes <= N_ONBOARD:
            raise ValueError(f"max_boxes must be 1..{N_ONBOARD}, got {self.max_boxes}")

        _check_int_range("n_zones_range", self.n_zones_range, minimum=0)
        _check_int_range("zone_size_range", self.zone_size_range, minimum=1)
        if self.n_zones_range[1] > 0 and self.zone_size_range[1] > self.grid_size:
            raise ValueError(f"zone_size_range {self.zone_size_range} does not fit the grid")
        if not isinstance(self.peak_windows, (tuple, list)):
            raise ValueError(f"peak_windows must be a sequence of (start, end) pairs, "
                             f"got {self.peak_windows!r}")
        for window in self.peak_windows:
            start, end = _pair("peak window", window)
            _check_number("peak window start", start, positive=False)
            _check_number("peak window end", end, positive=True)
            if not start < end:
                raise ValueError(f"peak window {window} must have start < end")

        for name, lam in self.loads.items():
            _check_number(f"load {name!r}", lam, positive=False)
        if not self.loads.get("high", 0) > 0:
            raise ValueError("loads must include 'high' > 0 (it scales the load feature)")


def _check_int(name: str, value) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")


def _pair(name: str, value):
    if not (isinstance(value, (tuple, list)) and len(value) == 2):
        raise ValueError(f"{name} must be a (min, max) pair, got {value!r}")
    return value


def _check_number(name: str, value, positive: bool) -> None:
    is_number = isinstance(value, numbers.Real) and not isinstance(value, bool)
    if not (is_number and math.isfinite(value) and (value > 0 if positive else value >= 0)):
        bound = "> 0" if positive else ">= 0"
        raise ValueError(f"{name} must be a finite number {bound}, got {value}")


def _check_int_range(name: str, pair, minimum: int) -> None:
    low, high = _pair(name, pair)
    _check_int(name, low)
    _check_int(name, high)
    if not minimum <= low <= high:
        raise ValueError(f"{name} must satisfy {minimum} <= min <= max, got {pair}")


def day_length(cfg: EnvConfig) -> float:
    """Latest time an order can still be open: release window + longest expiry."""
    return cfg.release_window + max(t.expiry_after for t in ORDER_TYPES)
