"""Unit fixtures only: no trained checkpoint, learned-policy episode or campaign."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from medical_uav.agents.networks import NETWORKS, split_obs
from medical_uav.agents.policy import load_checkpoint, save_checkpoint
from medical_uav.agents.ppo import RolloutBuffer, compute_gae, make_envs, ppo_update
from medical_uav.env import EnvConfig, MedicalUAVEnv
from medical_uav.env.actions import N_ACTIONS
from medical_uav.env.config import REWARDS
from medical_uav.env.observation import GLOBAL_DIM, OBS_DIM, observation_dim
from tests.conftest import make_order, make_sim


@pytest.mark.parametrize('name,penalty', [('v2_expiry03', .3), ('v2_expiry05', .5)])
def test_expiry_rewards_change_only_expiry(name, penalty, line_map):
    from medical_uav.env.observation import build_observation, rank_pending
    from medical_uav.env.masks import compute_action_mask

    base = EnvConfig(**REWARDS['v2'])
    cfg = EnvConfig(**REWARDS[name])
    assert cfg == replace(base, expiry_penalty=penalty)
    order = make_order(0, 0, 0, 1, .7, 0.)
    sim = make_sim(line_map, [order], cfg=cfg)
    sim.advance_to(60.)
    assert sim.total_reward == pytest.approx(-penalty)
    assert cfg.reward_on_time == 1 and cfg.reward_late == .1 and cfg.energy_cost == .001


def test_no_table_removes_only_table_in_nonempty_state(line_map):
    from medical_uav.env.observation import build_observation, rank_pending
    from medical_uav.env.masks import compute_action_mask

    orders = [make_order(i, i % 4, 0, i + 1, .4, 0.) for i in range(4)]
    sim = make_sim(line_map, orders)
    sim.do_pick(orders[0])
    slots = rank_pending(sim)
    mask = compute_action_mask(sim, slots)
    full = build_observation(sim, slots, mask)
    sim.cfg.include_travel_table = False
    short = build_observation(sim, slots, mask)
    assert full.shape == (452,) and short.shape == (403,)
    assert observation_dim(False) == 403 and observation_dim() == 452
    assert np.array_equal(short[:31], full[:31])
    assert np.array_equal(short[31:], full[80:])
    assert np.array_equal(mask, compute_action_mask(sim, slots))
    fg, fp, fc = split_obs(torch.tensor(full)[None])
    sg, sp, sc = split_obs(torch.tensor(short)[None])
    assert fg.shape == (1, 80) and sg.shape == (1, 31)
    assert torch.equal(sg, fg[:, :31]) and torch.equal(fp, sp) and torch.equal(fc, sc)


def test_no_table_env_space_and_multimap_reset():
    rng = np.random.default_rng(0)
    envs, state = make_envs(2, rng, EnvConfig(include_travel_table=False), [1000, 1001])
    assert state['obs'].shape == (2, 403)
    for env in envs:
        assert env.observation_space.shape == (403,)
        assert env.observation_space.contains(env._obs)
        obs, _ = env.reset(options={'demand_seed': 1234, 'map_seed': 1002})
        assert obs.shape == (403,) and not env.cfg.include_travel_table
        # A fixed WAIT checks step shape without running a learned policy.
        assert env.step(N_ACTIONS - 1)[0].shape == (403,)


@pytest.mark.parametrize('kind', list(NETWORKS))
def test_no_table_network_one_synthetic_ppo_update_and_roundtrip(kind, tmp_path):
    torch.manual_seed(8)
    net = NETWORKS[kind](obs_dim=403)
    obs = torch.rand(8, 403)
    mask = torch.ones(8, N_ACTIONS, dtype=torch.bool)
    mask[:, 0] = False
    # Set occupancy bits explicitly; arbitrary numeric slots still exercise all trainable heads.
    _, pending, cargo = split_obs(obs)
    pending[:, :, 0] = 1
    cargo[:, :, 0] = 1
    with torch.no_grad():
        logits, value = net(obs, mask)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
    assert logits.shape == (8, N_ACTIONS) and value.shape == (8,)
    shape = (4, 2)
    buf = RolloutBuffer(obs.numpy().reshape(*shape, 403), mask.numpy().reshape(*shape, N_ACTIONS),
                        actions.numpy().reshape(shape), dist.log_prob(actions).numpy().reshape(shape),
                        value.numpy().reshape(shape), np.arange(8, dtype=np.float32).reshape(shape)/8,
                        np.zeros(shape, bool), np.zeros(2, np.float32))
    adv, ret = compute_gae(buf.reward, buf.value, buf.done, buf.last_value)
    before = [p.detach().clone() for p in net.parameters()]
    stats = ppo_update(net, torch.optim.Adam(net.parameters(), lr=3e-4), buf, adv, ret,
                       epochs=1, minibatch=8, rng=np.random.default_rng(0))
    assert all(np.isfinite(v) for v in stats.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, net.parameters()))
    path = tmp_path / 'untrained_unit_fixture.pt'
    save_checkpoint(net, kind, path, {'fixture': True})
    loaded, info = load_checkpoint(path)
    assert loaded.obs_dim == 403 and info['fixture']
    with torch.no_grad():
        assert all(torch.equal(a, b) for a, b in zip(net(obs, mask), loaded(obs, mask)))


def test_legacy_full_table_checkpoint_layout(tmp_path):
    net = NETWORKS['set']()
    path = tmp_path / 'legacy_fixture.pt'
    save_checkpoint(net, 'set', path, {})
    data = torch.load(path)
    data.pop('obs_dim')
    torch.save(data, path)
    loaded, _ = load_checkpoint(path)
    assert loaded.obs_dim == OBS_DIM


def test_run_day_uses_checkpoint_layout_without_policy_execution(monkeypatch):
    import medical_uav.evaluation as evaluation
    from medical_uav.agents.policy import NetworkPolicy

    configs = []
    class StopFixture(Exception):
        pass
    def fake_env(config, load):
        configs.append(config)
        raise StopFixture
    monkeypatch.setattr(evaluation, 'MedicalUAVEnv', fake_env)
    policy = NetworkPolicy(NETWORKS['set'](obs_dim=403))
    with pytest.raises(StopFixture):
        evaluation.run_day(policy, 'medium', 100040, EnvConfig(map_seed=3))
    assert configs[0].map_seed == 3 and not configs[0].include_travel_table


def test_new_per_type_outcome_exports():
    from medical_uav.baselines.heuristics import slack_aware_policy
    from medical_uav.evaluation import heuristic, run_day
    from medical_uav.env.config import ORDER_TYPES

    row = run_day(heuristic(slack_aware_policy), 'medium', 100040)
    n = sum(row[f'released_{t.name}'] for t in ORDER_TYPES)
    served = sum(row[f'served_count_{t.name}'] for t in ORDER_TYPES)
    expired = sum(row[f'expired_count_{t.name}'] for t in ORDER_TYPES)
    assert served + expired == n
    assert row['served_rate'] == served/n and row['expiry_rate'] == expired/n


@pytest.mark.parametrize('reward', ['v2_expiry03', 'v2_expiry05'])
def test_train_cli_sizes_network_from_env_without_training(tmp_path, monkeypatch, reward):
    import importlib.util
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    path = Path(__file__).resolve().parents[1] / 'scripts/train_ppo.py'
    spec = importlib.util.spec_from_file_location('train_cli_fixture', path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    captured = {}
    def fake_envs(n, rng, config, maps):
        captured['config'], captured['maps'] = config, maps
        return [SimpleNamespace(observation_space=SimpleNamespace(shape=(403,)))], {}
    def fake_net(obs_dim):
        captured['obs_dim'] = obs_dim
        return torch.nn.Linear(obs_dim, 1)
    monkeypatch.setattr(script, 'make_envs', fake_envs)
    monkeypatch.setattr(script, 'NETWORKS', {'set': fake_net})
    monkeypatch.setattr(script, 'collect', lambda *a, **k: pytest.fail('No policy rollout authorized'))
    monkeypatch.setattr(script, 'evaluate_net', lambda *a, **k: pytest.fail('No evaluation authorized'))
    monkeypatch.setattr(sys, 'argv', ['train_ppo.py', '--net', 'set', '--maps', 'train',
                                     '--no-travel-table', '--reward', reward, '--decisions', '0',
                                     '--out', str(tmp_path/'mock_run')])
    script.main()
    assert captured['obs_dim'] == 403 and not captured['config'].include_travel_table
    assert captured['config'].expiry_penalty == REWARDS[reward]['expiry_penalty']
    assert captured['maps'] == list(range(1000, 1500))
