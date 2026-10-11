"""Discrete action table (§5)."""
from medical_uav.env.config import C2, M, NODE_NAMES

N_ACTIONS = 26
PICK0 = 0  # PICK(k), k = 0..11 -> pending slot k
DELIVER0 = 12  # DELIVER(n), n = 0..5 -> facility n
CHARGE0 = 18  # CHARGE(station, target), see CHARGE_TABLE
RETURN = 24
WAIT = 25

CHARGE_TABLE = ((M, 60.0), (M, 80.0), (M, 100.0), (C2, 60.0), (C2, 80.0), (C2, 100.0))


def decode(a: int):
    """Action index -> (kind, argument)."""
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
        return f"CHARGE({'C1' if station == M else 'C2'},{target:.0f})"
    return kind.upper()
