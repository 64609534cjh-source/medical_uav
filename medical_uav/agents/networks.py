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

    def act(self, observations, action_masks, deterministic=False):
        """Sample from legal actions, or select the legal argmax for evaluation."""
        logits, values = self(observations)
        distribution = masked_distribution(logits, action_masks)
        actions = distribution.probs.argmax(-1) if deterministic else distribution.sample()
        return actions, distribution.log_prob(actions), values

    def evaluate_actions(self, observations, action_masks, actions):
        """Recompute probabilities with the historical mask, never a live env mask."""
        logits, values = self(observations)
        if actions.shape != logits.shape[:-1] or actions.dtype != torch.int64:
            raise ValueError("actions must be int64 with one index per observation")
        if ((actions < 0) | (actions >= logits.shape[-1])).any():
            raise ValueError("action index outside the action space")
        distribution = masked_distribution(logits, action_masks)
        if not action_masks.gather(-1, actions.unsqueeze(-1)).all():
            raise ValueError("historical action is illegal under its saved mask")
        return distribution.log_prob(actions), distribution.entropy(), values


def masked_distribution(logits, action_masks):
    """Use the mask saved with these observations for every probability calculation."""
    if logits.shape != action_masks.shape or action_masks.dtype != torch.bool:
        raise ValueError("action_masks must be boolean and have the logits' shape")
    if not action_masks.any(dim=-1).all():
        raise ValueError("each observation must have at least one legal action")
    if not torch.isfinite(logits).all():
        raise ValueError("action logits must be finite")
    return Categorical(logits=logits.masked_fill(~action_masks, -torch.inf))
