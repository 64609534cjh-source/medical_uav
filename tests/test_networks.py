import numpy as np
import torch

from medical_uav.agents.networks import MLPNet, SetNet, count_parameters, masked_logits, split_obs
from medical_uav.env import MedicalUAVEnv
from medical_uav.env.observation import GLOBAL_DIM, OBS_DIM


def real_batch(n, load="medium", seed=100_000):
    """Return n consecutive observations and masks of one day run with random valid actions."""
    env = MedicalUAVEnv(load=load)
    obs, info = env.reset(options={"demand_seed": seed})
    rng = np.random.default_rng(0)
    observations, masks = [], []
    for _ in range(n):
        observations.append(obs)
        masks.append(info["action_mask"])
        obs, _, done, _, info = env.step(int(rng.choice(np.flatnonzero(info["action_mask"]))))
        if done:
            obs, info = env.reset(options={"demand_seed": seed + 1})
    return torch.as_tensor(np.array(observations)), torch.as_tensor(np.array(masks))


def test_split_obs_shapes():
    obs, _ = real_batch(2)
    assert obs.shape == (2, OBS_DIM)
    g, pending, cargo = split_obs(obs)
    assert g.shape == (2, GLOBAL_DIM) and pending.shape == (2, 12, 27) and cargo.shape == (2, 3, 16)
    rebuilt = torch.cat([g, pending.flatten(1), cargo.flatten(1)], dim=1)
    assert torch.equal(rebuilt, obs)


def test_masked_logits_zero_probability():
    logits = torch.randn(1, 26)
    mask = torch.zeros(1, 26, dtype=torch.bool)
    mask[0, [0, 13, 25]] = True
    p = torch.softmax(masked_logits(logits, mask), dim=1)
    assert p[~mask].abs().max() == 0.0
    assert torch.isclose(p.sum(), torch.tensor(1.0))


def overfit_loss(net, steps=300):
    """Train on 8 observations with fixed valid targets and return the final cross-entropy."""
    torch.manual_seed(0)
    obs, mask = real_batch(8)
    target = torch.tensor([int(np.flatnonzero(m.numpy())[-1]) for m in mask])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(steps):
        logits, _ = net(obs, mask)
        loss = torch.nn.functional.cross_entropy(logits, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item()


def check_shapes_and_mask(net):
    obs, mask = real_batch(5)
    logits, value = net(obs, mask)
    assert logits.shape == (5, 26) and value.shape == (5,)
    assert torch.softmax(logits, 1)[~mask].abs().max() == 0.0
    logits, value = net(obs[:1], mask[:1])
    assert logits.shape == (1, 26) and value.shape == (1,)


def test_mlp_shapes_mask_and_batch_of_one():
    torch.manual_seed(0)
    check_shapes_and_mask(MLPNet())


def test_mlp_parameter_count():
    expected = 2 * (452 * 256 + 256) + 2 * (256 * 256 + 256) + (256 * 26 + 26) + (256 + 1)
    assert count_parameters(MLPNet()) == expected


def test_mlp_overfits_tiny_batch():
    assert overfit_loss(MLPNet()) < 0.05


def state_with_orders(min_orders):
    """Return one observation and mask with at least min_orders occupied pending slots."""
    obs, mask = real_batch(120, load="high")
    occupied = obs[:, GLOBAL_DIM:GLOBAL_DIM + 12 * 27].reshape(-1, 12, 27)[:, :, 0].sum(1)
    i = int(torch.nonzero(occupied >= min_orders)[0])
    return obs[i:i + 1], mask[i:i + 1]


def test_set_shapes_mask_and_batch_of_one():
    torch.manual_seed(0)
    check_shapes_and_mask(SetNet())


def test_set_pending_permutation():
    torch.manual_seed(0)
    net = SetNet()
    obs, mask = state_with_orders(3)
    perm = torch.randperm(12)
    g, pending, cargo = split_obs(obs)
    obs_p = torch.cat([g, pending[:, perm].flatten(1), cargo.flatten(1)], dim=1)
    mask_p = mask.clone()
    mask_p[:, :12] = mask[:, :12][:, perm]
    with torch.no_grad():
        logits, value = net(obs, mask)
        logits_p, value_p = net(obs_p, mask_p)
    assert torch.allclose(logits_p[:, :12], logits[:, :12][:, perm], atol=1e-5)
    assert torch.allclose(logits_p[:, 12:], logits[:, 12:], atol=1e-5)
    assert torch.allclose(value_p, value, atol=1e-5)


def test_set_empty_and_single_order_finite():
    torch.manual_seed(0)
    net = SetNet()
    obs, mask = state_with_orders(1)
    g, pending, cargo = split_obs(obs)
    empty = torch.cat([g, torch.zeros_like(pending).flatten(1), torch.zeros_like(cargo).flatten(1)], 1)
    single_pending = torch.zeros_like(pending)
    first = int(torch.nonzero(pending[0, :, 0])[0])
    single_pending[0, first] = pending[0, first]
    single = torch.cat([g, single_pending.flatten(1), torch.zeros_like(cargo).flatten(1)], 1)
    only_wait = torch.zeros_like(mask)
    only_wait[0, 25] = True
    for o, m in ((empty, only_wait), (single, mask)):
        with torch.no_grad():
            logits, value = net(o, m)
        assert torch.isfinite(logits).all() and torch.isfinite(value).all()


def test_set_overfits_tiny_batch():
    assert overfit_loss(SetNet()) < 0.05
