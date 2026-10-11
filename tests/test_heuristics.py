import numpy as np
import pytest

from medical_uav.baselines.heuristics import charging_rule, edf_policy, random_policy, run_episode
from medical_uav.env import MedicalUAVEnv
from medical_uav.env.actions import CHARGE0, DELIVER0, WAIT


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
