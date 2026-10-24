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
    assert env.lam == pytest.approx(0.12)
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
        first_pick = next((a for a in range(12) if mask[a]), WAIT)
        a = next((a for a in range(12, 18) if mask[a]), first_pick)
        obs, r, term, trunc, info = env.step(a)
        kinds += [k for (_, k, _) in info["events"]]
        assert not term
    assert "pickup" in kinds and "deliver" in kinds


def test_step_after_done_raises():
    env = MedicalUAVEnv(load="low")
    rollout(env, 1)
    with pytest.raises(RuntimeError):
        env.step(WAIT)


def test_overload_episode_ends_with_every_order_delivered_or_expired():
    env = MedicalUAVEnv()
    rollout(env, 2, options={"lam": 0.4})
    statuses = [o.status for o in env.sim.orders]
    assert "expired" in statuses
    assert all(s in ("delivered", "expired") for s in statuses)


def test_reward_equals_sum_of_order_outcomes():
    env = MedicalUAVEnv(load="high")
    total, _ = rollout(env, 4)
    expected = 0.0
    for o in env.sim.orders:
        if o.status == "delivered":
            age = o.end_time - o.release
            expected += delivery_reward(age, o.limit, o.expiry_after, env.sim.cfg)
        else:
            expected -= 0.1  # expiry penalty (environment v2)
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


def test_same_seed_gives_identical_event_log():
    a, b = MedicalUAVEnv(), MedicalUAVEnv()
    rollout(a, 5)
    rollout(b, 5)
    assert a.sim.events == b.sim.events


def test_decision_log_records_actions():
    env = MedicalUAVEnv()
    env.reset(seed=0)
    env.step(WAIT)
    t0, name, reward = env.decision_log[0]
    assert t0 == 0.0 and name == "WAIT"
    env.reset(seed=1)
    assert env.decision_log == []


def test_decision_log_names_the_picked_order():
    env = MedicalUAVEnv()
    obs, info = env.reset(options={"demand_seed": 3})
    while not info["action_mask"][:12].any():
        obs, r, done, _, info = env.step(WAIT)
    order = env.slots[0]
    env.step(0)
    from medical_uav.env.config import NODE_NAMES, ORDER_TYPES

    expected = (f"PICK {ORDER_TYPES[order.type_idx].name} "
                f"{NODE_NAMES[order.origin]}->{NODE_NAMES[order.dest]}")
    assert env.decision_log[-1][1] == expected


def test_env_keeps_its_own_copy_of_the_config():
    from medical_uav.env import EnvConfig

    cfg = EnvConfig()
    env = MedicalUAVEnv(config=cfg)
    cfg.handling_min = 9.0
    cfg.loads["high"] = 1.0
    assert env.cfg.handling_min == 2.0 and env.cfg.loads["high"] == 0.12


@pytest.mark.parametrize("overrides", [
    {"handling_min": 3.0}, {"max_boxes": 2}, {"max_boxes": 1}, {"reserve": 25.0},
    {"min_charge_gain": 15.0}, {"payload_factor": 0.0, "charge_rate": 2.0},
])
def test_masks_match_execution_under_non_default_settings(overrides):
    # the simulator asserts that every masked-in action really completes (no load after expiry,
    # battery invariant kept); random rollouts on high load exercise expiries and disposal
    from medical_uav.env import EnvConfig

    env = MedicalUAVEnv(config=EnvConfig(**overrides), load="high")
    for seed in range(3):
        rollout(env, seed, rng_seed=seed)
