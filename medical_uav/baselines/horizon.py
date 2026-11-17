"""Rolling-horizon optimizer: at every decision, plan over the known orders, act, re-plan.

The planning copy holds what the optimizer knows (visible: what the agent observes; foresight:
also the hidden pending orders and the real orders of the next 60 minutes). Every sequence of
`depth` allowed actions is simulated exactly and finished by Slack-aware until all known work
is done; the first action of the best sequence is taken. Multistep lookahead with a base-policy
completion (Bertsekas, rollout algorithms), re-solved at each decision. Pre-registered in
docs/superpowers/specs/2026-10-27-rolling-horizon-preregistration.md.
"""
import copy

import numpy as np

from medical_uav.baselines.heuristics import slack_aware_policy
from medical_uav.baselines.scenario import _clone, planning_root

HORIZON = 60.0  # minutes of real future orders the foresight version sees


def foresight_root(env, horizon: float = HORIZON):
    """The planning copy plus every pending order and the orders released by t + horizon."""
    root = planning_root(env)
    real, sim = env.sim, root.sim
    sim.pending = [copy.copy(o) for o in real.pending]  # the slots rank them as the real ones
    future = [copy.copy(o) for o in real.orders[real._next_release:]
              if o.release <= real.t + horizon]
    sim.orders = sim.onboard + sim.pending + future
    sim._next_release = len(sim.onboard) + len(sim.pending)
    root._update_view()
    return root


def _work_left(sim) -> bool:
    return (sim._next_release < len(sim.orders) or bool(sim.pending)
            or any(o.status == "onboard" for o in sim.onboard))


def _after(env, action: int):
    c = _clone(env)
    c.step(action)
    return c


def _best(env, depth: int) -> float:
    """Best total reward from env: `depth` free actions, then Slack-aware finishes."""
    if not _work_left(env.sim):
        return env.sim.total_reward
    if depth == 0:
        c = _clone(env)
        while _work_left(c.sim):
            c.step(slack_aware_policy(c))
        return c.sim.total_reward
    return max(_best(_after(env, int(a)), depth - 1) for a in np.flatnonzero(env.action_masks()))


class RollingHorizon:
    """The optimizer as a policy(env); the real action mask is authoritative for the first
    action."""

    def __init__(self, depth: int = 3, foresight: bool = False, horizon: float = HORIZON):
        self.depth, self.foresight, self.horizon = depth, foresight, horizon

    def root(self, env):
        return foresight_root(env, self.horizon) if self.foresight else planning_root(env)

    def scores(self, env) -> dict[int, float]:
        """Value of each allowed action: the best plan found after taking it."""
        root = self.root(env)
        candidates = np.flatnonzero(env.action_masks())
        assert root.action_masks()[candidates].all(), "the planning copy disallows an action"
        return {int(a): _best(_after(root, int(a)), self.depth - 1) for a in candidates}

    def __call__(self, env) -> int:
        """The best-scoring action; among those within 1e-9 of the best, Slack-aware's choice
        (on the planning copy), else the lowest index."""
        scores = self.scores(env)
        top = max(scores.values())
        near = [a for a in sorted(scores) if scores[a] >= top - 1e-9]
        slack = slack_aware_policy(self.root(env))
        return slack if slack in near else near[0]
