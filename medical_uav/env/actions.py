"""Discrete action table: which actions exist and their indices.

Index layout: PICK(slot) x N_SLOTS | DELIVER(facility) | CHARGE(station, target) | RETURN | WAIT.
How long each action takes is decided by the simulator and EnvConfig, not here.
"""
from medical_uav.env.config import CHARGERS, FACILITIES, M, NODE_NAMES

N_SLOTS = 12  # pending orders the agent can see and PICK (also sizes the observation)
CHARGE_TARGETS = (60.0, 80.0, 100.0)  # battery % a CHARGE action fills up to
CHARGE_TABLE = tuple((station, target) for station in CHARGERS for target in CHARGE_TARGETS)

# DELIVER(n) uses the facility index n as its offset, so facilities must be 0..5 in order
assert FACILITIES == tuple(range(len(FACILITIES)))

PICK0 = 0
DELIVER0 = PICK0 + N_SLOTS
CHARGE0 = DELIVER0 + len(FACILITIES)
RETURN = CHARGE0 + len(CHARGE_TABLE)
WAIT = RETURN + 1
N_ACTIONS = WAIT + 1  # 12 + 6 + 6 + 1 + 1 = 26


def charge_action(station: int, target: float) -> int:
    """Index of CHARGE(station, target)."""
    return CHARGE0 + CHARGE_TABLE.index((station, target))


def decode(a: int):
    """Action index -> (kind, argument)."""
    if isinstance(a, bool) or not isinstance(a, int) or not 0 <= a < N_ACTIONS:
        raise ValueError(f"action must be an integer in 0..{N_ACTIONS - 1}, got {a!r}")
    if a < DELIVER0:
        return "pick", a - PICK0
    if a < CHARGE0:
        return "deliver", a - DELIVER0
    if a < RETURN:
        return "charge", CHARGE_TABLE[a - CHARGE0]
    if a == RETURN:
        return "return", None
    return "wait", None


def action_name(a: int) -> str:
    kind, arg = decode(a)
    if kind == "pick":
        return f"PICK({arg})"
    if kind == "deliver":
        return f"DELIVER({NODE_NAMES[arg]})"
    if kind == "charge":
        station, target = arg
        return f"CHARGE({'C1' if station == M else NODE_NAMES[station]},{target:.0f})"
    return kind.upper()
