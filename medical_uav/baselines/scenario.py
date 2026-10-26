"""Scenario rollout planner: Monte Carlo rollout over Slack-aware with sampled future arrivals.

At every decision it builds a planning copy holding only what the agent observes (time, place,
battery, cargo and the orders shown in the slots; no history, hidden or future orders), samples
S streams of future arrivals over the next 60 minutes from the demand model with its own random
streams, and for every allowed action and every stream runs the action and lets Slack-aware
finish all modelled work (no arrivals after the horizon). It takes the action with the best
mean total reward over the streams; ties go to Slack-aware's choice, then to the lower index.
All actions are scored on the same streams (common random numbers).

An adaptation of the rollout policy-improvement framework (Bertsekas) with sample-average action
values; pre-registered in docs/superpowers/specs/2026-10-27-final-phase-preregistration.md.
"""
import copy

import numpy as np

from medical_uav.baselines.heuristics import slack_aware_policy
from medical_uav.env import MedicalUAVEnv
from medical_uav.env.config import ORDER_TYPES
from medical_uav.env.masks import compute_action_mask
from medical_uav.env.observation import rank_pending
from medical_uav.env.orders import Order, _sample_route, _segments
from medical_uav.env.simulator import Simulator

HORIZON = 60.0  # minutes of sampled arrivals
MASTER_SEED = 271828
FUTURE_ID = 1_000_000  # ids of sampled orders, above any real order id


def sample_arrivals(rng, t: float, lam: float, cfg, first_id: int,
                    horizon: float = HORIZON) -> list[Order]:
    """Orders released in (t, min(t + horizon, release window)), drawn as generate_orders draws
    a day: Poisson counts per peak or off-peak segment, uniform times, then type, route, weight."""
    end = min(t + horizon, cfg.release_window)
    releases = []
    for a, b, mult in _segments(cfg):
        lo, hi = max(a, t), min(b, end)
        if hi > lo:
            releases.extend(rng.uniform(lo, hi, size=rng.poisson(lam * mult * (hi - lo))))
    releases.sort()
    shares = [ot.share for ot in ORDER_TYPES]
    orders = []
    for i, release in enumerate(releases):
        type_idx = int(rng.choice(len(ORDER_TYPES), p=shares))
        order_type = ORDER_TYPES[type_idx]
        origin, dest = _sample_route(rng, order_type.route)
        weight = float(rng.uniform(*order_type.weight_range))
        orders.append(Order(first_id + i, type_idx, origin, dest, weight, float(release),
                            order_type.limit, order_type.expiry_after))
    return orders


class _PlanningEnv(MedicalUAVEnv):
    """The environment without the observation vector, for fast planning steps."""

    def _update_view(self) -> None:
        self.slots = rank_pending(self.sim)
        self._mask = compute_action_mask(self.sim, self.slots)


def planning_root(env) -> _PlanningEnv:
    """A planning copy of what the agent observes: time, place, battery, cargo and the visible
    orders (copies with their ids, so the slots keep their order); no history."""
    root = object.__new__(_PlanningEnv)
    root.cfg, root.city_map, root.lam = env.cfg, env.city_map, env.lam
    root.action_space, root.decision_log, root._done = env.action_space, [], False
    root._obs = np.zeros(0, dtype=np.float32)
    real = env.sim
    sim = Simulator(env.cfg, env.city_map, [], env.lam)
    sim.t, sim.location, sim.battery = real.t, real.location, real.battery
    sim.onboard = [copy.copy(o) for o in real.onboard]
    sim.pending = [copy.copy(o) for o in env.slots]
    sim.orders = sim.onboard + sim.pending  # released; sampled arrivals are appended after them
    sim._next_release = len(sim.orders)
    root.sim = sim
    root._update_view()
    return root


def _clone(env: _PlanningEnv) -> _PlanningEnv:
    """Copy the mutable planning state; the map, settings and action space are shared."""
    return copy.deepcopy(env, {id(env.city_map): env.city_map, id(env.cfg): env.cfg,
                               id(env.action_space): env.action_space})


def _with_arrivals(root: _PlanningEnv, arrivals: list[Order]) -> _PlanningEnv:
    scenario = _clone(root)
    scenario.sim.orders = scenario.sim.orders + arrivals
    return scenario


def _rollout(scenario: _PlanningEnv, action: int) -> float:
    """Total reward of one scenario after `action`, with Slack-aware finishing all work."""
    c = _clone(scenario)
    c.step(action)
    sim = c.sim
    while (sim._next_release < len(sim.orders) or sim.pending
           or any(o.status == "onboard" for o in sim.onboard)):
        c.step(slack_aware_policy(c))
    return sim.total_reward


class ScenarioRollout:
    """The planner as a policy(env); its random streams depend only on its seed, the decision's
    index within the day and the scenario index, so scenarios 1-4 are shared by S = 4 and 16."""

    def __init__(self, scenarios: int = 4, horizon: float = HORIZON, seed: int = MASTER_SEED):
        self.scenarios, self.horizon, self.seed = scenarios, horizon, seed
        self.sim, self.decision = None, 0  # the day being served, decisions taken in it

    def arrivals(self, root, s: int) -> list[Order]:
        """Scenario s of the current decision."""
        rng = np.random.default_rng([self.seed, self.decision, s])
        return sample_arrivals(rng, root.sim.t, root.lam, root.cfg, FUTURE_ID, self.horizon)

    def scores(self, env) -> dict[int, float]:
        """Mean total reward over the scenarios of every allowed action."""
        if env.sim is not self.sim:  # a new day
            self.sim, self.decision = env.sim, 0
        root = planning_root(env)
        scenarios = [_with_arrivals(root, self.arrivals(root, s)) for s in range(self.scenarios)]
        self.decision += 1
        return {int(a): float(np.mean([_rollout(sc, int(a)) for sc in scenarios]))
                for a in np.flatnonzero(root.action_masks())}

    def __call__(self, env) -> int:
        scores = self.scores(env)
        best = slack_aware_policy(env)
        for a in sorted(scores):
            if scores[a] > scores[best] + 1e-9:
                best = a
        return best
