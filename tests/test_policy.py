import numpy as np
import torch

from medical_uav.agents.networks import MLPNet, SetNet
from medical_uav.agents.policy import NetworkPolicy, load_checkpoint, save_checkpoint
from medical_uav.env import MedicalUAVEnv
from medical_uav.env.actions import WAIT
from medical_uav.env.config import ENV_VERSION
from medical_uav.env.observation import OBS_DIM


def test_policy_chooses_valid_action():
    for net in (MLPNet(), SetNet()):
        policy = NetworkPolicy(net)
        env = MedicalUAVEnv(load="low")
        obs, info = env.reset(options={"demand_seed": 100_000})
        done = False
        while not done:
            a = policy(env, obs, info["action_mask"])
            assert info["action_mask"][a]
            obs, _, done, _, info = env.step(a)


def test_policy_only_wait_valid():
    policy = NetworkPolicy(SetNet())
    obs = MedicalUAVEnv().reset(options={"demand_seed": 100_000})[0]
    mask = np.zeros(26, dtype=bool)
    mask[WAIT] = True
    assert policy(None, obs, mask) == WAIT
    p = policy.probabilities(obs, mask)
    assert p.shape == (26,) and p[WAIT] == 1.0 and p.sum() == 1.0


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    net = SetNet()
    if torch.cuda.is_available():
        net = net.cuda()
    save_checkpoint(net, "set", tmp_path / "net.pt", {"note": "untrained"})
    loaded, info = load_checkpoint(tmp_path / "net.pt")
    obs = torch.randn(4, OBS_DIM)
    mask = torch.ones(4, 26, dtype=torch.bool)
    with torch.no_grad():
        a = net.cpu()(obs, mask)
        b = loaded(obs, mask)
    assert all(torch.allclose(x, y, atol=1e-6) for x, y in zip(a, b))
    assert info["note"] == "untrained" and info["env_version"] == ENV_VERSION and info["kind"] == "set"
