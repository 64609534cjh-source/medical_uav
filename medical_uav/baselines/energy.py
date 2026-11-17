"""Charging interventions of the energy study (pre-registered 2026-10-27).

Slack-100 is Slack-aware preferring full charges; PrePeakTopUp adds at most one full top-up
before each demand peak; without_full_charge hides the two CHARGE(*, 100) actions from a policy.
"""
from medical_uav.baselines.heuristics import slack_aware_policy
from medical_uav.env.actions import CHARGE0, CHARGE_TABLE, charge_action
from medical_uav.env.config import CHARGERS, EPS, FULL_BATTERY
from medical_uav.env.physics import charge_minutes

FULL_CHARGES = [CHARGE0 + i for i, (_, target) in enumerate(CHARGE_TABLE) if target == FULL_BATTERY]
PREPEAK_BELOW = 80.0  # battery % below which a top-up is taken
PREPEAK_WINDOW = 30.0  # minutes before a peak start in which it may be taken


def slack100_policy(env) -> int:
    """Slack-aware charging to 100 % rather than 80 % whenever it charges."""
    return slack_aware_policy(env, preference=(FULL_BATTERY, 80.0))


class PrePeakTopUp:
    """Slack-100 plus at most one full charge before each demand peak: in the 30 minutes before
    the peak, with nothing on board and the battery below 80 %, if a full charge at a station
    can finish by the peak start (the earliest finish is taken, ties to the lower node)."""

    def __init__(self):
        self.sim, self.used = None, set()  # the day being served, peaks already topped up for

    def __call__(self, env) -> int:
        if env.sim is not self.sim:  # a new day
            self.sim, self.used = env.sim, set()
        a = self.top_up(env)
        return a if a is not None else slack100_policy(env)

    def top_up(self, env) -> int | None:
        """A charge action before a coming peak when the battery is low and nothing is on board."""
        sim, mask = env.sim, env.action_masks()
        if sim.battery >= PREPEAK_BELOW or sim.onboard:  # onboard includes expired boxes
            return None
        for peak, _ in sim.cfg.peak_windows:
            if peak in self.used or not peak - PREPEAK_WINDOW <= sim.t < peak:
                continue
            options = []
            for station in CHARGERS:
                a = charge_action(station, FULL_BATTERY)
                if mask[a]:
                    arrival = sim.battery - sim.energy(sim.location, station, 0.0)
                    finish = (sim.t + sim.travel_time(sim.location, station)
                              + charge_minutes(arrival, FULL_BATTERY, sim.cfg))
                    if finish <= peak + EPS:
                        options.append((finish, station, a))
            if options:
                self.used.add(peak)
                return min(options)[2]
        return None


def without_full_charge(policy):
    """policy(env, obs, mask) that sees CHARGE(*, 100) as not allowed."""
    def masked(env, obs, mask):
        mask = mask.copy()
        mask[FULL_CHARGES] = False
        return policy(env, obs, mask)
    return masked
