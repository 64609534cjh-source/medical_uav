"""Flight, charging and reward formulas shared by the simulator and the baselines."""
from medical_uav.env.config import EPS, EnvConfig


def flight_time(dist_km: float, cfg: EnvConfig) -> float:
    return dist_km / cfg.speed_km_per_min


def flight_energy(dist_km: float, load_kg: float, cfg: EnvConfig) -> float:
    """Battery % used to fly dist_km with load_kg on board."""
    return dist_km * cfg.energy_per_km * (1.0 + cfg.payload_factor * load_kg)


def charge_minutes(battery: float, target: float, cfg: EnvConfig) -> float:
    return max(0.0, target - battery) / cfg.charge_rate


def delivery_reward(age: float, limit: float, expiry: float, cfg: EnvConfig) -> float:
    """+R if on time, then linear from +R at L down to -P at H."""
    if age <= limit + EPS:
        return cfg.reward_on_time
    late_fraction = (age - limit) / (expiry - limit)
    return cfg.reward_on_time - (cfg.reward_on_time + cfg.expiry_penalty) * late_fraction
