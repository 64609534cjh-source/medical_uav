"""Environment constants and tunable settings."""
from dataclasses import dataclass, field

# Fixed structure; changing it changes the observation and action spaces.

# Node indices. Charger C1 is located at the medical center M.
NODE_NAMES = ("M", "H1", "H2", "H3", "H4", "H5", "C2")
M, H1, H2, H3, H4, H5, C2 = range(len(NODE_NAMES))
N_NODES = len(NODE_NAMES)
FACILITIES = (M, H1, H2, H3, H4, H5)  # nodes where cargo is loaded and unloaded
HOSPITALS = (H1, H2, H3, H4, H5)
CHARGERS = (M, C2)

N_ONBOARD = 3  # onboard boxes listed in the observation; unused rows are zero-filled
FULL_BATTERY = 100.0  # battery is measured in percent
EPS = 1e-9  # tolerance for floating-point comparisons


@dataclass(frozen=True)
class OrderType:
    """Fixed properties of one order category."""

    name: str
    route: str  # center_to_hospital, hospital_to_center or hospital_to_hospital
    weight_range: tuple[float, float]  # kg, sampled uniformly
    limit: float  # minutes after release within which delivery is on time
    expiry_after: float  # minutes after release at which the order expires
    share: float  # fraction of all orders


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
    """Tunable environment settings."""

    # map
    grid_size: int = 16  # grid cells per side
    cell_km: float = 0.25  # side length of a grid cell
    n_zones_range: tuple[int, int] = (3, 3)  # (min, max) number of no-fly zones
    zone_size_range: tuple[int, int] = (2, 4)  # (min, max) zone side length in cells
    min_node_spacing_km: float = 0.75  # minimum distance between C2 and the other nodes
    facility_spacing_km: float = 1.4  # minimum distance between any two facilities (M and hospitals)
    c2_min_dist_from_m_km: float = 1.5  # minimum distance from C2 to M
    map_seed: int = 0  # seed for map generation
    max_map_tries: int = 100  # attempts before map generation gives up

    # drone
    speed_km_per_min: float = 1.0  # flight speed, independent of cargo
    energy_per_km: float = 3.5  # battery % used per km without cargo
    payload_factor: float = 0.25  # relative energy increase per kg of cargo
    reserve: float = 15.0  # battery % that must never be used
    charge_rate: float = 5.0  # battery % gained per minute of charging
    min_charge_gain: float = 5.0  # a CHARGE action must add more than this battery %
    handling_min: float = 2.0  # minutes to load, unload or dispose of one box
    max_boxes: int = 3  # cargo capacity in boxes
    max_weight: float = 2.0  # cargo capacity in kg

    # demand: the arrival rate is the base rate times the peak or off-peak multiplier
    release_window: float = 480.0  # minutes from the start of the day during which orders arrive
    # (start, end) of each peak period, in minutes from the start of the day
    peak_windows: tuple[tuple[float, float], ...] = ((60.0, 180.0), (300.0, 390.0))
    peak_multiplier: float = 2.0  # rate multiplier during peaks
    offpeak_multiplier: float = 0.6  # rate multiplier outside peaks
    # base arrival rate for each load level, in orders per minute
    loads: dict = field(default_factory=lambda: {"low": 0.04, "medium": 0.09, "high": 0.12})

    # actions
    wait_max: float = 10.0  # longest duration of one WAIT action, minutes

    # reward
    reward_on_time: float = 1.0  # reward per on-time delivery
    expiry_penalty: float = 0.5  # penalty per expired order
    energy_cost: float = 0.001  # penalty per battery % used

    def validate(self) -> None:
        """Raise ValueError for settings the observation layout cannot hold."""
        if not 1 <= self.max_boxes <= N_ONBOARD:
            raise ValueError(f"max_boxes must be 1..{N_ONBOARD}, got {self.max_boxes}")
        if self.loads.get("high", 0) <= 0:
            raise ValueError("loads must include 'high' > 0 (it scales the load feature)")


def day_length(cfg: EnvConfig) -> float:
    """Return the latest time an order can still be open: release window plus longest expiry."""
    return cfg.release_window + max(t.expiry_after for t in ORDER_TYPES)
