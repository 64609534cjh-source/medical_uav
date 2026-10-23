"""Policy/value networks. Each maps a batch of observations and action masks to
masked action logits [B, 26] and state values [B]."""
import torch

from medical_uav.env.actions import N_ACTIONS, N_SLOTS
from medical_uav.env.config import N_ONBOARD
from medical_uav.env.observation import GLOBAL_DIM, OBS_DIM, ONBOARD_DIM, PENDING_DIM

MASK_VALUE = -1e9  # logit given to actions the mask forbids


def split_obs(obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split observations [B, 452] into global [B, 80], pending [B, 12, 27] and cargo [B, 3, 16]."""
    pending_end = GLOBAL_DIM + N_SLOTS * PENDING_DIM
    g = obs[:, :GLOBAL_DIM]
    pending = obs[:, GLOBAL_DIM:pending_end].reshape(-1, N_SLOTS, PENDING_DIM)
    cargo = obs[:, pending_end:].reshape(-1, N_ONBOARD, ONBOARD_DIM)
    return g, pending, cargo


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return logits with forbidden actions (mask False) set to MASK_VALUE."""
    return logits.masked_fill(~mask, MASK_VALUE)


def count_parameters(net: torch.nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def _mlp(sizes: list[int]) -> torch.nn.Sequential:
    """Linear layers of the given sizes with tanh between them (none after the last)."""
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(torch.nn.Tanh())
    return torch.nn.Sequential(*layers)


class MLPNet(torch.nn.Module):
    """Reference network: separate actor and critic on the flat observation."""

    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 256, n_actions: int = N_ACTIONS):
        super().__init__()
        self.actor = _mlp([obs_dim, hidden, hidden, n_actions])
        self.critic = _mlp([obs_dim, hidden, hidden, 1])

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return masked_logits(self.actor(obs), mask), self.critic(obs).squeeze(-1)
