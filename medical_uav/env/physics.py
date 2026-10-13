"""Flight, charging and reward formulas shared by the simulator and the baselines."""
from medical_uav.env.config import EPS, EnvConfig


def flight_time(dist_km: float, cfg: EnvConfig) -> float:
    """Minutes to fly dist_km."""
    return dist_km / cfg.speed_km_per_min


def flight_energy(dist_km: float, load_kg: float, cfg: EnvConfig) -> float:
    """Battery % used to fly dist_km with load_kg on board."""
    return dist_km * cfg.energy_per_km * (1.0 + cfg.payload_factor * load_kg)


def charge_minutes(battery: float, target: float, cfg: EnvConfig) -> float:
    """Minutes to charge from battery % up to target %."""
    return max(0.0, target - battery) / cfg.charge_rate


def delivery_reward(age: float, limit: float, expiry_after: float, cfg: EnvConfig) -> float:
    """Reward for a box delivered `age` minutes after release.

    +R if age <= L (limit); then falls linearly to -P at H (expiry_after).
    """
    if age <= limit + EPS:
        return cfg.reward_on_time
    late_fraction = (age - limit) / (expiry_after - limit)
    return cfg.reward_on_time - (cfg.reward_on_time + cfg.expiry_penalty) * late_fraction
