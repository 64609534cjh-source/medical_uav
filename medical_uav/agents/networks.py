"""Small shared actor-critic and the environment's masked action distribution."""

import torch
from torch import nn
from torch.distributions import Categorical

from medical_uav.env.actions import N_ACTIONS, N_SLOTS
from medical_uav.env.observation import OBS_DIM, GLOBAL_DIM, PENDING_DIM, ONBOARD_DIM
from medical_uav.env.config import N_ONBOARD


class MaskedActorCritic(nn.Module):
    """Common historical-mask probability interface used by both architectures."""

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


class MLPActorCritic(MaskedActorCritic):
    """Encode one observation into 26 action logits and an unrestricted value."""

    architecture = "mlp_403_256_128_26"

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


def hidden_stack(input_dim, hidden_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(),
                         nn.Linear(hidden_dim, hidden_dim), nn.Tanh())


class StructuredActorCritic(MaskedActorCritic):
    """Shared order scoring, presence-masked means, and an invariant state value.

    Presence comes from the first feature in each slot, independently of action
    feasibility. In particular, real but unpickable orders still inform context.
    """

    architecture = "structured_mean_403_65232_26"

    def __init__(self):
        super().__init__()
        self.global_encoder = hidden_stack(GLOBAL_DIM, 64)
        self.pending_encoder = hidden_stack(PENDING_DIM, 64)
        self.cargo_encoder = hidden_stack(ONBOARD_DIM, 32)
        self.context_encoder = hidden_stack(160, 128)
        self.pick_head = nn.Sequential(nn.Linear(192, 64), nn.Tanh(), nn.Linear(64, 1))
        self.other_head = nn.Linear(128, N_ACTIONS - N_SLOTS)
        self.critic = nn.Linear(128, 1)

    @staticmethod
    def encode_set(rows, encoder):
        presence = (rows[..., 0] > 0).unsqueeze(-1)
        # Mask AFTER biased encoders; absent rows cannot contribute to the mean.
        embeddings = encoder(rows) * presence
        pooled = embeddings.sum(-2) / presence.sum(-2).clamp_min(1)
        return embeddings, pooled

    def forward(self, observations):
        leading = observations.shape[:-1]
        pending_end = GLOBAL_DIM + N_SLOTS * PENDING_DIM
        pending = observations[..., GLOBAL_DIM:pending_end].reshape(*leading, N_SLOTS, PENDING_DIM)
        cargo = observations[..., pending_end:].reshape(*leading, N_ONBOARD, ONBOARD_DIM)
        order_embeddings, order_pool = self.encode_set(pending, self.pending_encoder)
        _, cargo_pool = self.encode_set(cargo, self.cargo_encoder)
        global_features = self.global_encoder(observations[..., :GLOBAL_DIM])
        context = self.context_encoder(torch.cat((global_features, order_pool, cargo_pool), -1))
        expanded = context.unsqueeze(-2).expand(*leading, N_SLOTS, 128)
        picks = self.pick_head(torch.cat((expanded, order_embeddings), -1)).squeeze(-1)
        logits = torch.cat((picks, self.other_head(context)), -1)
        return logits, self.critic(context).squeeze(-1)


NETWORKS = {"mlp": MLPActorCritic, "structured": StructuredActorCritic,
            MLPActorCritic.architecture: MLPActorCritic,
            StructuredActorCritic.architecture: StructuredActorCritic}


def build_network(architecture="mlp"):
    if architecture not in NETWORKS:
        raise ValueError(f"unsupported network architecture: {architecture}")
    return NETWORKS[architecture]()
