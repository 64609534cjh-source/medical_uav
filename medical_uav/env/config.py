"""All constants of the environment in one place.

Every number here comes from the design spec
(docs/superpowers/specs/2026-10-10-medical-uav-design.md); the section is cited
next to each group so the code and the spec can be read side by side.
"""
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Nodes of the map (spec §4.1). Indices are fixed everywhere in the code.
# ---------------------------------------------------------------------------
M, H1, H2, H3, H4, H5, C2 = 0, 1, 2, 3, 4, 5, 6
NODE_NAMES = ("M", "H1", "H2", "H3", "H4", "H5", "C2")
N_NODES = 7
FACILITIES = (M, H1, H2, H3, H4, H5)  # places where boxes can be loaded/unloaded
HOSPITALS = (H1, H2, H3, H4, H5)
CHARGERS = (M, C2)  # charging station C1 is located at the center M

# Tolerance used for every floating-point time/energy comparison.
EPS = 1e-9


@dataclass(frozen=True)
class OrderType:
    """One kind of medical order (spec §4.3)."""

    name: str
    route: str  # "center_to_hospital", "hospital_to_center" or "hospital_to_hospital"
    weight_range: tuple[float, float]  # kg, sampled uniformly
    limit: float  # service limit L in minutes (on time if delivered by then)
    expiry: float  # expiry H in minutes (order is lost after this age)
    share: float  # probability that a new order has this type


ORDER_TYPES = (
    OrderType("plasma", "center_to_hospital", (0.6, 0.9), limit=30.0, expiry=60.0, share=0.20),
    OrderType("medicine", "center_to_hospital", (0.3, 0.6), limit=90.0, expiry=180.0, share=0.30),
    OrderType("sample", "hospital_to_center", (0.2, 0.4), limit=60.0, expiry=120.0, share=0.30),
    OrderType("supplies", "hospital_to_hospital", (0.4, 0.8), limit=75.0, expiry=150.0, share=0.20),
)


@dataclass
class EnvConfig:
    """Tunable parameters of the environment. Defaults are the spec values."""

    # --- map (§4.1) ---
    grid_size: int = 16  # cells per side
    cell_km: float = 0.25  # side length of one cell
    n_zones_range: tuple[int, int] = (2, 4)  # number of rectangular no-fly zones
    min_node_spacing_km: float = 0.75
    c2_min_dist_from_m_km: float = 1.0
    map_seed: int = 0  # the fixed map used for the main experiments
    max_map_tries: int = 100

    # --- drone (§4.2) ---
    speed_km_per_min: float = 1.0
    energy_per_km: float = 3.5  # battery % per km with no load
    payload_factor: float = 0.25  # +25 % energy per kg carried
    reserve: float = 15.0  # battery % that must never be used
    charge_rate: float = 5.0  # battery % per minute
    handling_min: float = 2.0  # minutes to load or unload one box
    max_boxes: int = 3
    max_weight: float = 2.0  # kg

    # --- demand (§4.3) ---
    release_window: float = 480.0  # orders are released during [0, 480) minutes
    peak_windows: tuple[tuple[float, float], ...] = ((60.0, 180.0), (300.0, 390.0))
    peak_multiplier: float = 2.0
    offpeak_multiplier: float = 0.6
    loads: dict = field(default_factory=lambda: {"low": 0.04, "medium": 0.07, "high": 0.10})
    pool_cap: int = 30  # max pending (not yet picked) orders; more are rejected

    # --- observation / actions (§5, §6) ---
    n_slots: int = 12  # pending orders visible to the policy
    wait_max: float = 10.0  # WAIT lasts at most this many minutes
    charge_targets: tuple[float, ...] = (60.0, 80.0, 100.0)

    # --- reward (§7) ---
    reward_on_time: float = 1.0  # R
    expiry_penalty: float = 0.5  # P
    energy_cost: float = 0.001  # per % of battery used in flight
