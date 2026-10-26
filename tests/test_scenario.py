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


def observable_view(env, mask=None):
    """An object holding only what the planner may read: time, place, battery, cargo, the
    visible orders, the arrival rate, the public settings and map, and the action mask."""
    from types import SimpleNamespace

    sim = SimpleNamespace(t=env.sim.t, location=env.sim.location, battery=env.sim.battery,
                          onboard=list(env.sim.onboard))
    mask = env.action_masks() if mask is None else mask
    return SimpleNamespace(sim=sim, slots=list(env.slots), lam=env.lam, cfg=env.cfg,
                           city_map=env.city_map, action_space=env.action_space,
                           action_masks=lambda: mask.copy())


def test_planner_needs_only_the_observable_view():
    from medical_uav.baselines.scenario import ScenarioRollout

    for k in range(5):
        env = busy_state(seed=100_000 + k, steps=30 + 5 * k)
        assert ScenarioRollout(4).scores(observable_view(env)) == ScenarioRollout(4).scores(env)
        assert ScenarioRollout(4)(observable_view(env)) == ScenarioRollout(4)(env)


def test_an_action_value_does_not_depend_on_the_other_candidates():
    from medical_uav.baselines.scenario import ScenarioRollout

    env = busy_state()
    full = ScenarioRollout(4).scores(env)
    some = env.action_masks()
    some[sorted(full)[::2]] = False  # drop every second candidate
    part = ScenarioRollout(4).scores(observable_view(env, some))
    assert part and all(part[a] == full[a] for a in part)


def test_ties_prefer_slack_then_the_lowest_index():
    from medical_uav.baselines.scenario import ScenarioRollout

    env = busy_state()
    slack = slack_aware_policy(env)
    low, high = [a for a in np.flatnonzero(env.action_masks()) if a != slack][:2]

    class Fixed(ScenarioRollout):
        def __init__(self, scores):
            super().__init__()
            self.fixed = scores

        def scores(self, env):
            return self.fixed

    # both others within 1e-9 of the best, Slack-aware not: the lower index wins
    assert Fixed({low: 0.9e-9, high: 1.8e-9, slack: 0.0})(env) == low
    assert Fixed({low: 1e-9, slack: 0.5e-9})(env) == slack  # Slack-aware within the tolerance


def test_planner_at_the_edges_of_the_day():
    from medical_uav.baselines.scenario import ScenarioRollout, planning_root

    # a box that expired on board (as if during a flight)
    env = next(e for e in (busy_state(seed=100_000 + k) for k in range(10)) if e.sim.onboard)
    env.sim.onboard[0].status = "expired"
    env._update_view()
    assert [o.status for o in planning_root(env).sim.onboard] == \
        [o.status for o in env.sim.onboard]
    scores = ScenarioRollout(4).scores(env)
    assert scores and all(np.isfinite(v) for v in scores.values())

    # the last minutes of the release window: arrivals stop at its end and rollouts finish
    env = MedicalUAVEnv(load="high")
    env.reset(options={"demand_seed": 100_003})
    while env.sim.t <= env.cfg.release_window - 15:
        env.step(slack_aware_policy(env))
    scores = ScenarioRollout(16).scores(env)
    assert scores and all(np.isfinite(v) for v in scores.values())
