"""Environment constants. Values follow the design spec (section numbers in comments)."""
from dataclasses import dataclass, field

# Node indices (§4.1). C1 is the charger at the center M.
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


# §4.3
ORDER_TYPES = (
    OrderType("plasma", "center_to_hospital", (0.6, 0.9), limit=30.0, expiry=60.0, share=0.20),
    OrderType("medicine", "center_to_hospital", (0.3, 0.6), limit=90.0, expiry=180.0, share=0.30),
    OrderType("sample", "hospital_to_center", (0.2, 0.4), limit=60.0, expiry=120.0, share=0.30),
    OrderType("supplies", "hospital_to_hospital", (0.4, 0.8), limit=75.0, expiry=150.0, share=0.20),
)


@dataclass
class EnvConfig:
    # map (§4.1)
    grid_size: int = 16
    cell_km: float = 0.25
    n_zones_range: tuple[int, int] = (2, 4)
    min_node_spacing_km: float = 0.75
    c2_min_dist_from_m_km: float = 1.0
    map_seed: int = 0
    max_map_tries: int = 100

    # drone (§4.2)
    speed_km_per_min: float = 1.0
    energy_per_km: float = 3.5  # battery % per km, empty
    payload_factor: float = 0.25  # +25% energy per kg
    reserve: float = 15.0
    charge_rate: float = 5.0  # % per min
    handling_min: float = 2.0  # per box, load or unload
    max_boxes: int = 3
    max_weight: float = 2.0

    # demand (§4.3)
    release_window: float = 480.0
    peak_windows: tuple[tuple[float, float], ...] = ((60.0, 180.0), (300.0, 390.0))
    peak_multiplier: float = 2.0
    offpeak_multiplier: float = 0.6
    loads: dict = field(default_factory=lambda: {"low": 0.04, "medium": 0.07, "high": 0.10})
    pool_cap: int = 30

    # actions / observation (§5, §6)
    n_slots: int = 12
    wait_max: float = 10.0
    charge_targets: tuple[float, ...] = (60.0, 80.0, 100.0)

    # reward (§7)
    reward_on_time: float = 1.0  # R
    expiry_penalty: float = 0.5  # P
    energy_cost: float = 0.001  # per % battery
