"""Small formulas for flight, charging and delivery reward.

Kept separate from the simulator so that baselines and the planner can reuse
exactly the same physics.
"""
from medical_uav.env.config import EPS, EnvConfig


def flight_time(dist_km: float, cfg: EnvConfig) -> float:
    """Minutes needed to fly `dist_km`. Speed does not depend on load (§4.2)."""
    return dist_km / cfg.speed_km_per_min


def flight_energy(dist_km: float, load_kg: float, cfg: EnvConfig) -> float:
    """Battery % used to fly `dist_km` carrying `load_kg` (§4.2).

    energy = distance * 3.5 %/km * (1 + 0.25 * load), so 2 kg costs 50 % more.
    """
    return dist_km * cfg.energy_per_km * (1.0 + cfg.payload_factor * load_kg)


def charge_minutes(battery: float, target: float, cfg: EnvConfig) -> float:
    """Minutes to charge linearly from `battery` up to `target` (0 if already there)."""
    return max(0.0, target - battery) / cfg.charge_rate


def delivery_reward(age: float, limit: float, expiry: float, cfg: EnvConfig) -> float:
    """Reward for a box delivered when the order is `age` minutes old (§7).

    - on time (age <= L): +R
    - late (L < age <= H): falls linearly from +R at L to -P at H
    """
    if age <= limit + EPS:
        return cfg.reward_on_time
    fraction_late = (age - limit) / (expiry - limit)
    return cfg.reward_on_time - (cfg.reward_on_time + cfg.expiry_penalty) * fraction_late
