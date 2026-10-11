import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from medical_uav.env import MedicalUAVEnv
from medical_uav.env.actions import WAIT
from medical_uav.env.physics import delivery_reward


def rollout(env, seed, rng_seed=0, options=None):
    rng = np.random.default_rng(rng_seed)
    obs, info = env.reset(seed=seed, options=options)
    total, steps = 0.0, 0
    while True:
        valid = np.flatnonzero(info["action_mask"])
        obs, r, term, trunc, info = env.step(int(rng.choice(valid)))
        env.sim.check_invariant()
        assert info["action_mask"].any()
        assert env.observation_space.contains(obs)
        total += r
        steps += 1
        if term or trunc:
            return total, steps


def test_check_env():
    check_env(MedicalUAVEnv(), skip_render_check=True)


def test_determinism_same_seed():
    assert rollout(MedicalUAVEnv(), 3) == rollout(MedicalUAVEnv(), 3)


def test_demand_seed_and_load_options():
    env = MedicalUAVEnv()
    env.reset(options={"demand_seed": 7, "load": "high"})
    n_high = len(env.sim.orders)
    assert env.lam == pytest.approx(0.10)
    env.reset(options={"demand_seed": 7, "load": "low"})
    assert env.lam == pytest.approx(0.04) and len(env.sim.orders) < n_high


@pytest.mark.parametrize("seed", range(10))
def test_random_rollouts_keep_invariant(seed):
    rollout(MedicalUAVEnv(load="high"), seed)


def test_invalid_action_becomes_wait():
    env = MedicalUAVEnv()
    obs, info = env.reset(seed=0)
    invalid = int(np.flatnonzero(~info["action_mask"])[0])
    t0 = env.sim.t
    _, _, _, _, info = env.step(invalid)
    assert info["invalid_action"] and info["action_name"] == "WAIT" and env.sim.t > t0


def test_info_events_report_delivery():
    env = MedicalUAVEnv(load="medium")
    obs, info = env.reset(seed=0)
    kinds = []
    while "deliver" not in kinds:
        mask = info["action_mask"]
        a = next((a for a in range(12, 18) if mask[a]), next((a for a in range(12) if mask[a]), WAIT))
        obs, r, term, trunc, info = env.step(a)
        kinds += [k for (_, k, _) in info["events"]]
        assert not term
    assert "pickup" in kinds and "deliver" in kinds


def test_step_after_done_raises():
    env = MedicalUAVEnv(load="low")
    rollout(env, 1)
    with pytest.raises(RuntimeError):
        env.step(WAIT)


def test_overload_episode_runs_and_counts_rejections():
    env = MedicalUAVEnv()
    rollout(env, 2, options={"lam": 0.4})
    statuses = [o.status for o in env.sim.orders]
    assert "rejected" in statuses
    assert all(s in ("delivered", "expired", "rejected") for s in statuses)


def test_reward_equals_sum_of_order_outcomes():
    env = MedicalUAVEnv(load="high")
    total, _ = rollout(env, 4)
    expected = 0.0
    for o in env.sim.orders:
        if o.status == "delivered":
            expected += delivery_reward(o.end_time - o.release, o.limit, o.expiry_after, env.sim.cfg)
        else:
            expected -= 0.5
    expected -= 0.001 * env.sim.energy_used
    assert total == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("action", [-1, 26])
def test_out_of_range_action_raises(action):
    env = MedicalUAVEnv()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(action)


def test_default_demand_seed_in_training_range():
    env = MedicalUAVEnv()
    for seed in range(20):
        env.reset(seed=seed)
        assert 0 <= env.demand_seed < 100_000
