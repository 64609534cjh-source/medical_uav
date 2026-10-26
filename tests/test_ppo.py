import numpy as np
import torch

from medical_uav.agents.networks import SetNet
from medical_uav.agents.ppo import collect, compute_gae, make_envs, ppo_update


def test_gae_toy_trajectory():
    # one env, 4 steps, the episode ends after step 1 (index 1)
    rewards = np.array([[1.0], [0.0], [2.0], [1.0]])
    values = np.array([[0.5], [0.4], [0.3], [0.2]])
    dones = np.array([[False], [True], [False], [False]])
    adv, ret = compute_gae(rewards, values, dones, last_value=np.array([0.6]), lam=0.95)
    # by hand, gamma = 1: t3: 1 + 0.6 - 0.2 = 1.4; t2: 2 + 0.2 - 0.3 + 0.95 * 1.4 = 3.23;
    # t1 (done): 0 - 0.4 = -0.4; t0: 1 + 0.4 - 0.5 + 0.95 * -0.4 = 0.52
    assert np.allclose(adv[:, 0], [0.52, -0.4, 3.23, 1.4])
    assert np.allclose(ret[:, 0], [1.02, 0.0, 3.53, 1.6])


def rollout(n_envs=2, n_steps=32):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    net = SetNet()
    envs, state = make_envs(n_envs, rng)
    return net, collect(envs, net, state, n_steps, rng)


def recomputed_log_probs(net, buf):
    obs = torch.as_tensor(buf.obs.reshape(-1, buf.obs.shape[-1]))
    mask = torch.as_tensor(buf.mask.reshape(-1, buf.mask.shape[-1]))
    with torch.no_grad():
        logits, _ = net(obs, mask)
    dist = torch.distributions.Categorical(logits=logits)
    return dist, dist.log_prob(torch.as_tensor(buf.action.reshape(-1))).numpy()


def test_ratio_one_before_update():
    net, buf = rollout()
    _, log_prob = recomputed_log_probs(net, buf)
    assert np.allclose(log_prob, buf.log_prob.reshape(-1), atol=1e-5)


def test_masked_actions_never_sampled():
    net, buf = rollout(n_steps=64)
    flat_mask = buf.mask.reshape(-1, 26)
    flat_action = buf.action.reshape(-1)
    assert flat_mask[np.arange(len(flat_action)), flat_action].all()
    dist, _ = recomputed_log_probs(net, buf)
    invalid = torch.as_tensor(np.argmin(flat_mask, axis=1))  # first masked action of each state
    has_invalid = torch.as_tensor(~flat_mask.all(axis=1))
    assert (dist.log_prob(invalid)[has_invalid] <= -1e8).all()


def test_update_changes_parameters_and_is_finite():
    net, buf = rollout()
    adv, ret = compute_gae(buf.reward, buf.value, buf.done, buf.last_value)
    before = [p.detach().clone() for p in net.parameters()]
    opt = torch.optim.Adam(net.parameters(), lr=3e-4, eps=1e-5)
    stats = ppo_update(net, opt, buf, adv, ret, ent_coef=0.01, rng=np.random.default_rng(0))
    assert all(np.isfinite(v) for v in stats.values())
    assert any(not torch.equal(b, p) for b, p in zip(before, net.parameters()))


def test_train_ppo_smoke(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    from medical_uav.agents.policy import load_checkpoint

    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(repo / "scripts" / "train_ppo.py"), "--net", "mlp",
                             "--decisions", "4096", "--eval-every", "4096", "--eval-days", "1",
                             "--out", str(tmp_path / "run")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "run" / "updates.csv").exists() and (tmp_path / "run" / "evals.csv").exists()
    checkpoints = sorted((tmp_path / "run").glob("ckpt_*.pt"))
    assert len(checkpoints) == 1
    _, info = load_checkpoint(checkpoints[0])
    assert info["decisions"] == 4096 and info["seed"] == 11
    import json
    meta = json.loads((tmp_path / "run" / "run.json").read_text())
    assert {"env_version", "git_revision", "net", "seed", "reward"} <= set(meta)


def test_ppo_days_avoid_demonstration_seeds():
    from medical_uav.agents.ppo import _new_day
    from medical_uav.env import MedicalUAVEnv
    from medical_uav.splits import DEMO_SEEDS, RL_SEED_RANGE

    env, rng = MedicalUAVEnv(), np.random.default_rng(11)
    seeds = []
    for _ in range(500):
        _new_day(env, rng)
        seeds.append(env.demand_seed)
    lo, hi = RL_SEED_RANGE
    assert all(lo <= s < hi for s in seeds) and not set(seeds) & set(DEMO_SEEDS)


def test_make_envs_uses_the_reward_variant():
    from medical_uav.env.config import REWARDS, EnvConfig

    envs, _ = make_envs(2, np.random.default_rng(0), EnvConfig(**REWARDS["m1"]))
    assert all(env.cfg.late_reward == "linear" and env.cfg.expiry_penalty == 0.5 for env in envs)


def fixed_batch():
    net, buf = rollout(n_envs=2, n_steps=16)
    obs = torch.as_tensor(buf.obs.reshape(-1, buf.obs.shape[-1]))
    mask = torch.as_tensor(buf.mask.reshape(-1, 26))
    return obs, mask


def test_warm_up_keeps_the_policy():
    from medical_uav.agents.ppo import warm_up_value

    torch.manual_seed(0)
    net = SetNet()
    obs, mask = fixed_batch()
    with torch.no_grad():
        before = torch.softmax(net(obs, mask)[0], 1)
    envs, state = make_envs(2, np.random.default_rng(1))
    warm_up_value(net, envs, state, np.random.default_rng(1), rollouts=2, epochs=1, n_steps=32)
    with torch.no_grad():
        after = torch.softmax(net(obs, mask)[0], 1)
    assert torch.allclose(before, after, atol=1e-6)


def test_day_returns_cover_finished_days_only():
    from medical_uav.agents.ppo import RolloutBuffer, day_returns

    # two envs, 4 steps; env 0 finishes a day after step 1, env 1 never finishes
    reward = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0]], np.float32)
    done = np.array([[False, False], [True, False], [False, False], [False, False]])
    obs = np.arange(8, dtype=np.float32).reshape(4, 2, 1)
    buf = RolloutBuffer(obs=obs, mask=np.ones((4, 2, 1), bool), action=np.zeros((4, 2)),
                        log_prob=np.zeros((4, 2)), value=np.zeros((4, 2)), reward=reward,
                        done=done, last_value=np.zeros(2))
    obs, mask, ret = day_returns(buf)
    assert obs[:, 0].tolist() == [0.0, 2.0]  # env 0, steps 0 and 1
    assert ret.tolist() == [3.0, 2.0]


def test_warm_up_brings_values_closer_to_day_returns():
    from medical_uav.agents.ppo import warm_up_value

    torch.manual_seed(0)
    net = SetNet()
    envs, state = make_envs(2, np.random.default_rng(1))
    stats = warm_up_value(net, envs, state, np.random.default_rng(1), rollouts=4, epochs=4,
                          n_steps=64, check_steps=400)
    assert stats["value_error_after"] < stats["value_error_before"]
    assert stats["decisions"] == 2 * (4 * 64 + 400)


def test_train_ppo_from_a_bc_checkpoint(tmp_path):
    import json

    from medical_uav.agents.policy import load_checkpoint
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(repo / "scripts" / "train_ppo.py"), "--net", "set",
                             "--init", str(repo / "checkpoints" / "bc_set.pt"),
                             "--warmup-rollouts", "1", "--decisions", "2048", "--eval-every",
                             "2048", "--eval-days", "1", "--out", str(tmp_path / "run")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    meta = json.loads((tmp_path / "run" / "run.json").read_text())
    assert meta["init"].endswith("bc_set.pt") and meta["warmup"]["decisions"] == 16 * (128 + 512)
    _, info = load_checkpoint(next((tmp_path / "run").glob("ckpt_*.pt")))
    assert info["reward"] == "v2" and info["init"].endswith("bc_set.pt") and info["git_revision"]
