import copy

import numpy as np
import pytest

from medical_uav.baselines.heuristics import slack_aware_policy
from medical_uav.env import EnvConfig, MedicalUAVEnv
from medical_uav.env.orders import demand_multiplier


def busy_state(seed=100_000, steps=40, load="high"):
    env = MedicalUAVEnv(load=load)
    env.reset(options={"demand_seed": seed})
    for _ in range(steps):
        env.step(slack_aware_policy(env))
    return env


def test_sample_arrivals_follow_the_demand_model():
    from medical_uav.baselines.scenario import sample_arrivals

    cfg, lam, t = EnvConfig(), 0.12, 40.0  # the window (40, 100) spans the start of a peak
    expected = lam * sum(demand_multiplier(x + 0.5, cfg) for x in np.arange(t, t + 60))
    rng = np.random.default_rng(0)
    draws = [sample_arrivals(rng, t, lam, cfg, first_id=1000) for _ in range(2000)]
    assert np.mean([len(d) for d in draws]) == pytest.approx(expected, rel=0.03)
    for d in draws[:200]:
        releases = [o.release for o in d]
        assert releases == sorted(releases) and all(t < r < t + 60 for r in releases)
        assert [o.id for o in d] == list(range(1000, 1000 + len(d)))
        assert all(o.status == "future" for o in d)
    late = sample_arrivals(rng, cfg.release_window - 10, lam, cfg, first_id=0)
    assert all(o.release < cfg.release_window for o in late)


def test_planning_root_holds_only_the_observable_state():
    from medical_uav.baselines.scenario import planning_root

    env = busy_state()
    before = (env.sim.t, len(env.sim.orders), len(env.sim.segments), env.sim.total_reward)
    root = planning_root(env)
    assert [o.id for o in root.slots] == [o.id for o in env.slots]
    assert (root.action_masks() == env.action_masks()).all()
    assert (root.sim.t, root.sim.location, root.sim.battery) == \
        (env.sim.t, env.sim.location, env.sim.battery)
    assert {o.id for o in root.sim.pending} == {o.id for o in env.slots}
    assert [o.id for o in root.sim.onboard] == [o.id for o in env.sim.onboard]
    assert all(o.status in ("pending", "onboard", "expired") for o in root.sim.orders)
    assert root.sim.segments == [] and root.sim.total_reward == 0.0
    assert all(a is not b for a, b in zip(root.sim.onboard, env.sim.onboard))  # copies
    assert before == (env.sim.t, len(env.sim.orders), len(env.sim.segments), env.sim.total_reward)


def test_scenarios_one_to_four_are_shared_by_4_and_16():
    from medical_uav.baselines.scenario import ScenarioRollout, planning_root

    root = planning_root(busy_state())
    small, large = ScenarioRollout(4), ScenarioRollout(16)
    for s in range(4):
        a, b = small.arrivals(root, s), large.arrivals(root, s)
        assert [(o.release, o.type_idx, o.origin, o.dest, o.weight) for o in a] == \
            [(o.release, o.type_idx, o.origin, o.dest, o.weight) for o in b]
    assert small.arrivals(root, 0) != small.arrivals(root, 1)


def overflowing_states(n):
    """n distinct mid-day states at a very high rate, many with more orders than slots."""
    states = []
    for k in range(n):
        env = MedicalUAVEnv()
        env.reset(options={"demand_seed": 100_000 + k % 20, "lam": 0.2})
        for _ in range(20 + 8 * (k // 20)):
            env.step(slack_aware_policy(env))
        states.append(env)
    return states


def test_planner_ignores_hidden_and_future_information():
    from medical_uav.baselines.scenario import ScenarioRollout

    rng = np.random.default_rng(0)
    states = overflowing_states(100)
    assert len({(env.demand_seed, env.sim.t) for env in states}) == 100
    assert sum(len(env.sim.pending) > len(env.slots) for env in states) >= 50
    for env in states:
        before = ScenarioRollout(4).scores(env)
        other = copy.deepcopy(env)
        sim = other.sim
        shown = {o.id for o in other.slots}
        hidden = [o for o in sim.pending if o.id not in shown]
        for o in sim.orders[sim._next_release:] + hidden:
            o.dest = int(rng.choice([d for d in range(6) if d != o.origin]))
            o.weight = float(rng.uniform(0.2, 0.9))
            o.limit, o.expiry_after = o.limit * 3, o.expiry_after * 3
        for o in sim.orders[sim._next_release:]:
            o.release = sim.t + 0.5
        removed = {o.id for o in hidden[::2]}
        sim.pending = [o for o in sim.pending if o.id not in removed]
        other.demand_seed = 12345
        other.np_random = np.random.default_rng(99)
        assert ScenarioRollout(4).scores(other) == before


def test_without_arrivals_the_values_are_the_visible_rollout():
    from medical_uav.baselines.rollout import rollout_value
    from medical_uav.baselines.scenario import ScenarioRollout

    for env in [busy_state(seed=100_000 + k, steps=30 + 5 * k, load="medium") for k in range(5)]:
        scores = ScenarioRollout(4, horizon=0.0).scores(env)
        start = env.sim.total_reward
        for a, value in scores.items():
            assert value == pytest.approx(rollout_value(env, a) - start, abs=1e-9)


def test_planner_choice_is_at_least_slack_on_its_own_scenarios():
    from medical_uav.baselines.scenario import ScenarioRollout

    for k in range(5):
        env = busy_state(seed=100_000 + k, steps=30 + 5 * k, load="medium")
        scores = ScenarioRollout(4).scores(env)
        chosen = ScenarioRollout(4)(env)
        assert scores[chosen] >= scores[slack_aware_policy(env)] - 1e-9
        assert env.action_masks()[chosen]


def test_planner_runs_a_day_with_valid_actions_and_leaves_the_env_alone():
    from medical_uav.baselines.scenario import ScenarioRollout

    planner = ScenarioRollout(4)
    env = MedicalUAVEnv(load="low")
    obs, info = env.reset(options={"demand_seed": 100_000})
    done = False
    while not done:
        state = (env.sim.t, env.sim.battery, len(env.sim.orders), env.sim._next_release,
                 [o.id for o in env.sim.pending], env.np_random.bit_generator.state["state"])
        a = planner(env)
        assert state == (env.sim.t, env.sim.battery, len(env.sim.orders), env.sim._next_release,
                         [o.id for o in env.sim.pending],
                         env.np_random.bit_generator.state["state"])
        assert info["action_mask"][a]
        obs, _, done, _, info = env.step(a)
    assert planner.decision == len(env.decision_log)
