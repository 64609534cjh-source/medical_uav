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
    # map
    grid_size: int = 16
    cell_km: float = 0.25
    n_zones_range: tuple[int, int] = (2, 4)
    min_node_spacing_km: float = 0.75
    c2_min_dist_from_m_km: float = 1.0
    map_seed: int = 0
    max_map_tries: int = 100

    # drone
    speed_km_per_min: float = 1.0
    energy_per_km: float = 3.5  # battery % per km, empty
    payload_factor: float = 0.25  # +25% energy per kg
    reserve: float = 15.0
    charge_rate: float = 5.0  # % per min
    handling_min: float = 2.0  # per box, load or unload
    max_boxes: int = 3
    max_weight: float = 2.0

    # demand
    release_window: float = 480.0
    peak_windows: tuple[tuple[float, float], ...] = ((60.0, 180.0), (300.0, 390.0))
    peak_multiplier: float = 2.0
    offpeak_multiplier: float = 0.6
    loads: dict = field(default_factory=lambda: {"low": 0.04, "medium": 0.08, "high": 0.12})
    pool_cap: int = 30

    # actions / observation
    n_slots: int = 12
    wait_max: float = 10.0

    # reward
    reward_on_time: float = 1.0  # R
    expiry_penalty: float = 0.5  # P
    energy_cost: float = 0.001  # per % battery

    def validate(self) -> None:
        """Raise ValueError for settings the environment cannot run with."""
        for name in ["grid_size", "map_seed", "max_map_tries", "max_boxes", "pool_cap", "n_slots"]:
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
        if self.n_slots != 12:
            raise ValueError(f"n_slots must be 12 (fixed observation layout), got {self.n_slots}")
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
