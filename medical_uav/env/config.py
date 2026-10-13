"""Environment constants and their validation."""
import math
from dataclasses import dataclass, field

# Node indices. C1 is the charger at the center M.
M, H1, H2, H3, H4, H5, C2 = 0, 1, 2, 3, 4, 5, 6
NODE_NAMES = ("M", "H1", "H2", "H3", "H4", "H5", "C2")
N_NODES = 7
FACILITIES = (M, H1, H2, H3, H4, H5)
HOSPITALS = (H1, H2, H3, H4, H5)
CHARGERS = (M, C2)

EPS = 1e-9


@dataclass(frozen=True)
class OrderType:
    name: str
    route: str  # center_to_hospital / hospital_to_center / hospital_to_hospital
    weight_range: tuple[float, float]  # kg
    limit: float  # service limit L (min)
    expiry: float  # expiry H (min)
    share: float


ORDER_TYPES = (
    OrderType("plasma", "center_to_hospital", (0.6, 0.9), limit=30.0, expiry=60.0, share=0.20),
    OrderType("medicine", "center_to_hospital", (0.3, 0.6), limit=90.0, expiry=180.0, share=0.30),
    OrderType("sample", "hospital_to_center", (0.2, 0.4), limit=60.0, expiry=120.0, share=0.30),
    OrderType("supplies", "hospital_to_hospital", (0.4, 0.8), limit=75.0, expiry=150.0, share=0.20),
)


@dataclass
class EnvConfig:
    """All tunable settings. Defaults are the main experiment; override fields for ablations."""

    # map: a grid_size x grid_size grid of cells, each cell_km wide (16 x 0.25 = 4 km square)
    grid_size: int = 16
    cell_km: float = 0.25
    n_zones_range: tuple[int, int] = (2, 4)  # number of rectangular no-fly zones (min, max)
    min_node_spacing_km: float = 0.75  # minimum distance between any two nodes
    c2_min_dist_from_m_km: float = 1.0  # keeps the second charger away from the center
    map_seed: int = 0  # random seed of the map; same seed -> same map (main experiments use 0)
    max_map_tries: int = 100  # give up if no valid map is found after this many random attempts

    # drone
    speed_km_per_min: float = 1.0  # 60 km/h, the same with or without cargo
    energy_per_km: float = 3.5  # battery % used per km when empty
    payload_factor: float = 0.25  # each kg of cargo adds 25 % to the energy per km
    reserve: float = 15.0  # battery % that must always remain (masks enforce it)
    charge_rate: float = 5.0  # battery % gained per minute of charging
    handling_min: float = 2.0  # minutes to load or to unload one box
    max_boxes: int = 3  # cargo capacity in boxes ...
    max_weight: float = 2.0  # ... and in kg; both limits apply

    # demand: orders are released during [0, release_window) minutes (08:00-16:00).
    # Arrival rate at time t = base rate (one of `loads`) x multiplier:
    # peak_multiplier inside peak_windows, offpeak_multiplier elsewhere.
    release_window: float = 480.0
    # peaks at 09:00-11:00 and 13:00-14:30
    peak_windows: tuple[tuple[float, float], ...] = ((60.0, 180.0), (300.0, 390.0))
    peak_multiplier: float = 2.0
    offpeak_multiplier: float = 0.6
    # base rate in orders per minute; expected orders per day = 582 x rate (about 23 / 47 / 70)
    loads: dict = field(default_factory=lambda: {"low": 0.04, "medium": 0.08, "high": 0.12})
    pool_cap: int = 30  # max orders waiting to be picked up; further arrivals are rejected

    # actions
    wait_max: float = 10.0  # WAIT ends at the next release or expiry, at the latest after this

    # reward
    reward_on_time: float = 1.0  # R: per box delivered on time
    expiry_penalty: float = 0.5  # P: per order that expires or is rejected
    energy_cost: float = 0.001  # per % of battery used in flight

    def validate(self) -> None:
        """Raise ValueError for settings the environment cannot run with."""
        for name in ["grid_size", "map_seed", "max_map_tries", "max_boxes", "pool_cap"]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer, got {value!r}")
        positive = ["grid_size", "cell_km", "speed_km_per_min", "energy_per_km", "charge_rate",
                    "handling_min", "max_weight", "release_window", "wait_max", "max_map_tries"]
        for name in positive:
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be a finite number > 0, got {value}")
        for name in ["payload_factor", "expiry_penalty", "energy_cost", "reward_on_time"]:
            value = getattr(self, name)
            if not (math.isfinite(value) and value >= 0):
                raise ValueError(f"{name} must be a finite number >= 0, got {value}")
        if not (math.isfinite(self.reserve) and 0 <= self.reserve < 100):
            raise ValueError(f"reserve must be in [0, 100), got {self.reserve}")
        if not 1 <= self.max_boxes <= 3:
            raise ValueError(f"max_boxes must be 1..3 (3 onboard slots), got {self.max_boxes}")
        if self.pool_cap < 1:
            raise ValueError(f"pool_cap must be >= 1, got {self.pool_cap}")
        for name, lam in self.loads.items():
            if not (math.isfinite(lam) and lam >= 0):
                raise ValueError(f"load {name!r} must be a finite rate >= 0, got {lam}")
        if not self.loads.get("high", 0) > 0:
            raise ValueError("loads must include 'high' > 0 (it scales the load feature)")


def day_length(cfg: EnvConfig) -> float:
    """Latest time an order can still be open: release window + longest expiry."""
    return cfg.release_window + max(t.expiry for t in ORDER_TYPES)
