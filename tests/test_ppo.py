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
