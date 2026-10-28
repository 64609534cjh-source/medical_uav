"""Visible-order rollout planner: a one-step lookahead over Slack-aware.

At every decision it tries each allowed action on a copy of the day that holds only what the
agent can observe (the cargo and the up to 12 orders shown in the slots; no hidden or future
orders), lets Slack-aware finish every order in the copy, and takes the action whose copy ends
with the highest total reward (the environment's own reward). Ties go to Slack-aware's choice.
"""
import copy

import numpy as np

from medical_uav.baselines.heuristics import slack_aware_policy


def visible_copy(env):
    """Return a copy of env that keeps the cargo and the orders shown in the slots, and drops
    pending orders outside the slots and orders not yet released."""
    c = copy.deepcopy(env, {id(env.city_map): env.city_map, id(env.cfg): env.cfg})
    sim = c.sim
    shown = {o.id for o in c.slots}
    sim.pending = [o for o in sim.pending if o.id in shown]
    sim.orders = [o for o in sim.orders[:sim._next_release]
                  if o.status != "pending" or o.id in shown]
    sim._next_release = len(sim.orders)
    c._update_view()  # the same orders rank the same, so the slots keep their order
    return c


def rollout_value(env, action: int) -> float:
    """Total reward of a visible copy after taking `action` and letting Slack-aware finish."""
    c = visible_copy(env)
    c.step(action)
    while c.sim.pending or any(o.status == "onboard" for o in c.sim.onboard):
        c.step(slack_aware_policy(c))
    return c.sim.total_reward


def rollout_policy(env) -> int:
    """Take the allowed action with the best rollout value; ties go to Slack-aware's choice."""
    best_action = slack_aware_policy(env)
    best_value = rollout_value(env, best_action)
    for a in np.flatnonzero(env.action_masks()):
        if a != best_action:
            value = rollout_value(env, int(a))
            if value > best_value + 1e-9:
                best_action, best_value = int(a), value
    return best_action
