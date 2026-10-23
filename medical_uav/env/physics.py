"""Flight, charging and reward formulas shared by the simulator and the baselines."""
from medical_uav.env.config import EPS, EnvConfig


def flight_time(dist_km: float, cfg: EnvConfig) -> float:
    """Return the minutes needed to fly dist_km."""
    return dist_km / cfg.speed_km_per_min


def flight_energy(dist_km: float, load_kg: float, cfg: EnvConfig) -> float:
    """Return the battery % used to fly dist_km carrying load_kg."""
    return dist_km * cfg.energy_per_km * (1.0 + cfg.payload_factor * load_kg)


def charge_minutes(battery: float, target: float, cfg: EnvConfig) -> float:
    """Return the minutes needed to charge from battery to target (both in %)."""
    return (target - battery) / cfg.charge_rate


def delivery_reward(age: float, limit: float, cfg: EnvConfig) -> float:
    """Return the reward for a box delivered `age` minutes after release: the on-time reward up
    to `limit`, the smaller late reward after it (an order past its expiry is never delivered)."""
    return cfg.reward_on_time if age <= limit + EPS else cfg.reward_late
