import copy

import numpy as np

from medical_uav.baselines.heuristics import slack_aware_policy
from medical_uav.baselines.rollout import rollout_policy, rollout_value, visible_copy
from medical_uav.env import MedicalUAVEnv


def mid_day_states(n, load="high"):
    """n environments stopped at busy moments of screen days, after Slack-aware decisions."""
    states = []
    for k in range(n):
        env = MedicalUAVEnv(load=load)
        env.reset(options={"demand_seed": 100_000 + k % 5})
        for _ in range(30 + 3 * k):  # a high-load day has about 110 decisions
            env.step(slack_aware_policy(env))
        states.append(env)
    return states


def test_visible_copy_hides_future_and_hidden_orders():
    env = mid_day_states(1)[0]
    c = visible_copy(env)
    visible = [o.id for o in env.slots]
    assert sorted(o.id for o in c.sim.pending) == sorted(visible)
    assert [o.id for o in c.slots] == visible
    assert all(o.status != "future" for o in c.sim.orders)
    assert env.sim.pending and len(env.sim.orders) > len(c.sim.orders)  # the original is untouched


def overflowing_states(n):
    """n mid-day states at a very high load, where more orders wait than the 12 slots show."""
    states = []
    for k in range(n):
        env = MedicalUAVEnv()
        env.reset(options={"demand_seed": 100_000 + k % 20, "lam": 0.2})
        for _ in range(20 + 8 * (k // 20)):  # 20 days x 5 stopping points
            env.step(slack_aware_policy(env))
        states.append(env)
    return states


def test_rollout_ignores_future_and_hidden_orders():
    rng = np.random.default_rng(0)
    states = overflowing_states(100)
    assert len({(env.demand_seed, env.sim.t) for env in states}) == 100  # all distinct
    assert sum(len(env.sim.pending) > len(env.slots) for env in states) >= 50  # hidden orders exist
    for env in states:
        before = rollout_policy(env)
        other = copy.deepcopy(env)
        sim = other.sim
        shown = {o.id for o in other.slots}
        hidden = [o for o in sim.pending if o.id not in shown]
        for o in sim.orders[sim._next_release:] + hidden:
            o.dest = int(rng.choice([d for d in range(6) if d != o.origin]))
            o.weight = float(rng.uniform(0.2, 0.9))
            o.limit, o.expiry_after = o.limit * 3, o.expiry_after * 3  # later deadlines and expiry
        for o in sim.orders[sim._next_release:]:
            o.release = sim.t + 0.5  # future orders would arrive almost at once
        removed = {o.id for o in hidden[::2]}  # and half of the hidden orders disappear
        sim.pending = [o for o in sim.pending if o.id not in removed]
        assert rollout_policy(other) == before


def test_rollout_only_valid_actions():
    env = MedicalUAVEnv(load="low")
    obs, info = env.reset(options={"demand_seed": 100_000})
    done = False
    while not done:
        a = rollout_policy(env)
        assert info["action_mask"][a]
        obs, _, done, _, info = env.step(a)


def test_rollout_at_least_as_good_as_slack_on_its_own_score():
    for env in mid_day_states(10, load="medium"):
        chosen = rollout_policy(env)
        assert rollout_value(env, chosen) >= rollout_value(env, slack_aware_policy(env)) - 1e-9
