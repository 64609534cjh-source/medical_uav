"""Action indices, in order: PICK(slot) x N_SLOTS, DELIVER(facility), CHARGE(station, target),
RETURN, WAIT."""
import numbers

from medical_uav.env.config import CHARGERS, FACILITIES, FULL_BATTERY, M, NODE_NAMES

N_SLOTS = 12  # pending orders shown to the agent
CHARGE_TARGETS = (60.0, 80.0, FULL_BATTERY)  # charge up to this battery %
CHARGE_TABLE = tuple((station, target) for station in CHARGERS for target in CHARGE_TARGETS)

# DELIVER index = DELIVER0 + facility, so facilities must be numbered 0..5
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
    if isinstance(a, bool) or not isinstance(a, numbers.Integral) or not 0 <= a < N_ACTIONS:
        raise ValueError(f"action must be an integer in 0..{N_ACTIONS - 1}, got {a!r}")
    a = int(a)
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
