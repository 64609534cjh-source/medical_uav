import numpy as np
import pytest

from medical_uav.baselines.heuristics import (charging_rule, edf_policy, random_policy, run_episode,
                                              slack_aware_policy)
from medical_uav.env import MedicalUAVEnv
from medical_uav.env.actions import CHARGE0, DELIVER0, PICK0, WAIT
from medical_uav.env.config import M
from medical_uav.env.masks import compute_action_mask
from medical_uav.env.observation import rank_pending
from tests.conftest import make_order, make_sim


def test_edf_picks_earliest_deadline():
    env = MedicalUAVEnv()
    env.reset(seed=11)
    while not env.slots:
        env.step(WAIT)
    a = edf_policy(env)
    mask = env.action_masks()
    valid_picks = [k for k in range(len(env.slots)) if mask[k]]
    assert a < DELIVER0 and valid_picks
    assert env.slots[a].deadline == min(env.slots[k].deadline for k in valid_picks)


@pytest.mark.parametrize("seed", range(5))
def test_policies_only_choose_valid_actions(seed):
    env = MedicalUAVEnv(load="high")
    obs, info = env.reset(seed=seed)
    done = False
    while not done:
        a = edf_policy(env)
        assert info["action_mask"][a]
        c = charging_rule(env, 50.0)
        assert c is None or info["action_mask"][c]
        obs, r, done, _, info = env.step(a)


def test_charging_rule_low_battery_charges():
    env = MedicalUAVEnv()
    env.reset(seed=0)
    env.sim.battery = 30.0
    env._update_view()
    a = charging_rule(env, 40.0)
    assert a is not None and CHARGE0 <= a < CHARGE0 + 6


def test_metrics_account_for_every_order():
    m = run_episode(MedicalUAVEnv(), edf_policy, 0)
    assert 0.0 <= m["on_time_rate"] <= m["served_rate"] <= 1.0
    total = m["served_rate"] + m["expiry_rate"] + m["rejected"] / max(m["n_orders"], 1)
    assert total == pytest.approx(1.0)
    assert 0.0 < m["busy_share"] <= 1.0 and 0.0 <= m["charge_share"] < 1.0


def test_edf_beats_random_on_low_load():
    env = MedicalUAVEnv(load="low")
    rng = np.random.default_rng(0)
    edf = np.mean([run_episode(env, edf_policy, s)["on_time_rate"] for s in range(5)])
    rnd = np.mean([run_episode(env, lambda e: random_policy(e, rng), s)["on_time_rate"]
                   for s in range(5)])
    assert edf > rnd


def test_run_episode_seed_is_the_demand_seed():
    env = MedicalUAVEnv()
    run_episode(env, edf_policy, 3)
    assert env.demand_seed == 3


class SimEnv:
    """Minimal env-like wrapper around a hand-built simulator."""

    def __init__(self, sim):
        self.sim = sim
        self.slots = rank_pending(sim)

    def action_masks(self):
        return compute_action_mask(self.sim, self.slots)


def test_slack_aware_delivers_here_first(line_map):
    a = make_order(0, 0, M, 2, 0.6, 0.0)
    sim = make_sim(line_map, [a])
    sim.do_pick(a)
    sim.fly_to(2)  # at H2 with the box for H2 still on board
    sim.battery = 35.0  # below theta=40: charging would be wanted
    assert slack_aware_policy(SimEnv(sim), theta=40.0) == DELIVER0 + 2


def test_slack_aware_batches_same_destination(line_map):
    a, c = make_order(0, 1, M, 2, 0.5, 0.0), make_order(1, 1, M, 2, 0.5, 0.0)
    sim = make_sim(line_map, [a, c])
    sim.do_pick(a)  # at M, box for H2 on board, c waits at M for H2
    env = SimEnv(sim)
    assert slack_aware_policy(env) == PICK0 + env.slots.index(c)
    assert edf_policy(env) == DELIVER0 + 2  # EDF delivers first on the deadline tie


def test_slack_aware_prefers_feasible_over_hopeless(line_map):
    late = make_order(0, 0, M, 1, 0.7, 0.0)  # deadline 30: hopeless at t=45
    fresh = make_order(1, 1, M, 2, 0.5, 40.0)  # deadline 130
    sim = make_sim(line_map, [late, fresh])
    sim.advance_to(45.0)
    env = SimEnv(sim)
    assert slack_aware_policy(env) == PICK0 + env.slots.index(fresh)
    assert edf_policy(env) == PICK0 + env.slots.index(late)


@pytest.mark.parametrize("lam", [0.04, 0.08, 0.12, 0.4])
def test_slack_aware_only_valid_actions(lam):
    env = MedicalUAVEnv()
    obs, info = env.reset(options={"demand_seed": 1, "lam": lam})
    done = False
    while not done:
        a = slack_aware_policy(env)
        assert info["action_mask"][a]
        obs, r, done, _, info = env.step(a)


def test_slack_aware_beats_edf_on_medium():
    env = MedicalUAVEnv()
    opts = {"lam": 0.08}
    def on_time(policy):
        return np.mean([run_episode(env, policy, s, opts)["on_time_rate"] for s in range(10)])

    slack, edf = on_time(slack_aware_policy), on_time(edf_policy)
    assert slack > edf + 0.1


def test_slack_aware_delivery_counts_if_any_box_can_be_on_time(line_map):
    late = make_order(0, 0, M, 3, 0.6, 0.0)  # plasma to H3, deadline 30: late at t=45
    fresh = make_order(1, 1, M, 3, 0.5, 5.0)  # medicine to H3, deadline 95
    other = make_order(2, 1, M, 1, 0.4, 10.0)  # waiting at M for H1, deadline 100
    sim = make_sim(line_map, [late, fresh, other])
    sim.do_pick(late)
    sim.advance_to(5.0)
    sim.do_pick(fresh)
    sim.advance_to(45.0)
    # DELIVER(H3): arrive 49, late box 49-51, fresh box 51-53 <= 95 -> still worth it now
    assert slack_aware_policy(SimEnv(sim)) == DELIVER0 + 3
