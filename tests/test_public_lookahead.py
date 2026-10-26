"""No live-state access, paired futures, macro horizon and strict fallback."""
import copy
import time

import numpy as np
import pytest
import torch

from medical_uav.agents.networks import build_network
from medical_uav.baselines.heuristics import slack_aware_policy
from medical_uav.env import MedicalUAVEnv
from medical_uav.env.config import EnvConfig
from medical_uav.research.lookahead import PublicLookaheadPolicy, PublicSnapshot, Ineligible, clone_public_world
from medical_uav.env.orders import generate_orders


def state_with_work():
    env = MedicalUAVEnv(load='high')
    obs, info = env.reset(options={'demand_seed': 7, 'load': 'high'})
    for _ in range(100):
        if obs[31:355].reshape(12, 27)[:, 0].sum():
            return env, torch.from_numpy(obs), torch.from_numpy(info['action_mask'])
        obs, _, done, _, info = env.step(slack_aware_policy(env))
        assert not done
    raise AssertionError('no public work state')


def test_public_roundtrip_and_independent_future_do_not_touch_live_state():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    env, obs, mask = state_with_work()
    model = build_network('destination').eval()
    snapshot = PublicSnapshot(obs.numpy(), mask.numpy(), env.cfg, env.city_map)
    world = snapshot.build_world([])
    np.testing.assert_allclose(world._obs, obs.numpy(), atol=5e-5, rtol=1e-5)
    assert np.array_equal(world.action_masks(), mask.numpy())
    before = copy.deepcopy(env.sim.__dict__)
    first = PublicLookaheadPolicy(model, scenarios=2, horizon=5, budget_ms=None, planner_seed=9).eval()
    action1 = first.act(obs, mask, True)[0]
    assert mask[action1]
    assert env.sim.t == before['t'] and env.sim.total_reward == before['total_reward']
    assert [(o.id, o.status, o.weight, o.release) for o in env.sim.orders] == [
        (o.id, o.status, o.weight, o.release) for o in before['orders']]
    # Keep public inputs fixed; poison private unreleased orders and day ID.
    # The planner cannot inspect either because its API accepts only obs/mask.
    for order in env.sim.orders:
        if order.status == 'future':
            order.release += 900
            order.weight += 10
    env.demand_seed = 99999
    second = PublicLookaheadPolicy(model, scenarios=2, horizon=5, budget_ms=None, planner_seed=9).eval()
    action2 = second.act(obs, mask, True)[0]
    assert action1 == action2
    assert first.last_scenario_signatures == second.last_scenario_signatures
    assert len(first.last_scenario_signatures) == 2
    assert first.last_scores == second.last_scores
    assert first.counts['completed_comparisons'] == 1
    env.close()


def test_hidden_pending_or_unrepresented_public_history_is_ineligible():
    env, obs, mask = state_with_work()
    hidden = obs.numpy().copy()
    hidden[25] = 1/30
    with pytest.raises(Ineligible, match='hidden_pending_orders'):
        PublicSnapshot(hidden, mask.numpy(), env.cfg, env.city_map)
    missing = obs.numpy().copy()
    missing[30] += .1
    with pytest.raises(Ineligible, match='unrepresented_recent_arrival_history'):
        PublicSnapshot(missing, mask.numpy(), env.cfg, env.city_map)
    env.close()


def test_zero_time_budget_returns_exact_frozen_action_and_no_simulation():
    torch.manual_seed(18)
    env, obs, mask = state_with_work()
    model = build_network('destination').eval()
    policy = PublicLookaheadPolicy(model, budget_ms=0).eval()
    assert policy.act(obs, mask, True)[0] == model.act(obs, mask, True)[0]
    assert policy.counts['fallback_budget'] == 1
    assert policy.counts['extra_simulator_transitions'] == 0
    env.close()


def test_fast_synthetic_clone_matches_full_copy_scores_and_isolates_order_state():
    torch.set_num_threads(1)
    torch.manual_seed(21)
    env, obs, mask = state_with_work()
    model = build_network('destination').eval()
    policy = PublicLookaheadPolicy(model, scenarios=2, horizon=10, budget_ms=None).eval()
    snapshot = PublicSnapshot(obs.numpy(), mask.numpy(), env.cfg, env.city_map)
    future = [o for o in generate_orders(71, snapshot.lam, env.cfg) if o.release > snapshot.t]
    for i, order in enumerate(future):
        order.id = len(snapshot.current)+i
    world = snapshot.build_world(future)
    untouched = [vars(o).copy() for o in world.sim.orders]
    for action in torch.where(mask)[0].tolist():
        full, fast = copy.deepcopy(world), clone_public_world(world)
        assert fast.city_map is world.city_map and fast.sim.city_map is world.sim.city_map
        assert fast.sim.orders[0] is not world.sim.orders[0]
        assert all(any(o is candidate for candidate in fast.sim.orders) for o in fast.slots)
        score_full = policy._score(full, action, time.perf_counter())
        score_fast = policy._score(fast, action, time.perf_counter())
        assert score_fast == score_full
        assert fast.sim.t == full.sim.t and fast.sim.battery == full.sim.battery
        assert [vars(o) for o in fast.sim.orders] == [vars(o) for o in full.sim.orders]
    assert [vars(o) for o in world.sim.orders] == untouched
    with pytest.raises(ValueError, match='public-derived'):
        clone_public_world(env)
    env.close()


def test_macro_crossing_horizon_finishes_and_terminal_tail_is_zero():
    torch.manual_seed(19)
    env, obs, mask = state_with_work()
    model = build_network('destination').eval()
    snapshot = PublicSnapshot(obs.numpy(), mask.numpy(), env.cfg, env.city_map)
    # Use a public-derived synthetic world and close arrivals before the action.
    world = snapshot.build_world([])
    world.cfg.release_window = world.sim.t
    world.sim.cfg.release_window = world.sim.t
    action = next(a for a in range(12) if world.action_masks()[a])
    expected = copy.deepcopy(world)
    expected.step(action)
    policy = PublicLookaheadPolicy(model, scenarios=1, horizon=.001, budget_ms=None).eval()
    score = policy._score(world, action, time.perf_counter())
    assert world.sim.t == expected.sim.t
    assert world.sim.t > snapshot.t+.001
    if world.sim.is_done():
        assert score == pytest.approx(world.sim.total_reward)
    else:
        assert score == pytest.approx(world.sim.total_reward + float(model(torch.from_numpy(world._obs))[1].detach()))
    # WAIT until the one released order expires: terminal must omit any value.
    terminal = snapshot.build_world([])
    terminal.cfg.release_window = terminal.sim.t
    terminal.sim.cfg.release_window = terminal.sim.t
    policy.horizon = 500
    score = policy._score(terminal, 25, time.perf_counter())
    assert terminal.sim.is_done()
    assert score == pytest.approx(terminal.sim.total_reward)
    env.close()
