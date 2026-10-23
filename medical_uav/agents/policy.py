"""A trained network as a policy, and saving/loading networks with their metadata."""
from pathlib import Path

import numpy as np
import torch

from medical_uav.agents.networks import NETWORKS
from medical_uav.env.config import ENV_VERSION


def save_checkpoint(net: torch.nn.Module, kind: str, path, info: dict) -> None:
    """Save the network weights with its kind ("mlp" or "set"), the environment version and info."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.cpu() for k, v in net.state_dict().items()}
    torch.save({"kind": kind, "env_version": ENV_VERSION, "state_dict": state, "info": info}, path)


def load_checkpoint(path) -> tuple[torch.nn.Module, dict]:
    """Load a network on the CPU in evaluation mode; return it and its metadata."""
    ckpt = torch.load(path, map_location="cpu")
    net = NETWORKS[ckpt["kind"]]()
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, {**ckpt["info"], "kind": ckpt["kind"], "env_version": ckpt["env_version"]}


class NetworkPolicy:
    """Greedy policy: the valid action with the highest logit. Uses only the observation and mask."""

    def __init__(self, net: torch.nn.Module):
        self.net = net

    def _logits(self, obs: np.ndarray, mask: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            logits, _ = self.net(torch.as_tensor(obs, dtype=torch.float32)[None],
                                 torch.as_tensor(mask, dtype=torch.bool)[None])
        return logits[0]

    def __call__(self, env, obs: np.ndarray, mask: np.ndarray) -> int:
        return int(self._logits(obs, mask).argmax())

    def probabilities(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Return the network's probability for each of the 26 actions (0 for invalid ones)."""
        return torch.softmax(self._logits(obs, mask), dim=0).numpy()
