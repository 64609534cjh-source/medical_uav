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


def delivery_reward(age: float, limit: float, expiry_after: float, cfg: EnvConfig) -> float:
    """Return the reward for a box delivered `age` minutes after release: the on-time reward up
    to `limit`; after it the late reward, or with late_reward "linear" (the M1 reward) a value
    falling from the on-time reward to minus the expiry penalty at `expiry_after`."""
    if age <= limit + EPS:
        return cfg.reward_on_time
    if cfg.late_reward == "linear":
        late_fraction = (age - limit) / (expiry_after - limit)
        return cfg.reward_on_time - (cfg.reward_on_time + cfg.expiry_penalty) * late_fraction
    return cfg.reward_late
