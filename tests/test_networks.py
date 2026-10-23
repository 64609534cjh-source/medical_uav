import numpy as np
import torch

from medical_uav.agents.networks import masked_logits, split_obs
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
