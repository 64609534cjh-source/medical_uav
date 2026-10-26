"""Small shared actor-critic and the environment's masked action distribution."""

import torch
from torch import nn
from torch.distributions import Categorical

from medical_uav.env.actions import N_ACTIONS
from medical_uav.env.observation import OBS_DIM


class MLPActorCritic(nn.Module):
    """Encode one observation into 26 action logits and an unrestricted value."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(OBS_DIM, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
        )
        self.actor = nn.Linear(128, N_ACTIONS)
        self.critic = nn.Linear(128, 1)

    def forward(self, observations):
        features = self.encoder(observations)
        return self.actor(features), self.critic(features).squeeze(-1)


def masked_distribution(logits, action_masks):
    """Use the mask saved with these observations for every probability calculation."""
    if logits.shape != action_masks.shape or action_masks.dtype != torch.bool:
        raise ValueError("action_masks must be boolean and have the logits' shape")
    if not action_masks.any(dim=-1).all():
        raise ValueError("each observation must have at least one legal action")
    if not torch.isfinite(logits).all():
        raise ValueError("action logits must be finite")
    return Categorical(logits=logits.masked_fill(~action_masks, -torch.inf))
