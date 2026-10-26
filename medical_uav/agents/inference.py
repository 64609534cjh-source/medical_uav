"""Optional inference-only reuse; the training architecture and weight keys stay fixed."""

import torch
from torch.nn import functional as F

from medical_uav.agents.networks import StructuredActorCritic
from medical_uav.env.actions import N_SLOTS
from medical_uav.env.config import N_ONBOARD
from medical_uav.env.observation import GLOBAL_DIM, PENDING_DIM, ONBOARD_DIM


class FactoredStructuredInference(StructuredActorCritic):
    """Compute the shared PICK context projection once, reusing original weights.

    Floating-point reordering can slightly change PICK logits. Deployment argmax
    falls back to the historical path near a tie; no additional parameters exist.
    """

    def forward(self, observations):
        leading = observations.shape[:-1]
        end = GLOBAL_DIM + N_SLOTS * PENDING_DIM
        pending = observations[..., GLOBAL_DIM:end].reshape(*leading, N_SLOTS, PENDING_DIM)
        cargo = observations[..., end:].reshape(*leading, N_ONBOARD, ONBOARD_DIM)
        orders, order_pool = self.encode_set(pending, self.pending_encoder)
        _, cargo_pool = self.encode_set(cargo, self.cargo_encoder)
        glob = self.global_encoder(observations[..., :GLOBAL_DIM])
        context = self.context_encoder(torch.cat((glob, order_pool, cargo_pool), -1))
        first, _, last = self.pick_head
        # W[c,e] + b = Wc*c + We*e + b. Context's 128×64 product is
        # shared over the 12 rows instead of recomputed for each candidate.
        context_projection = F.linear(context, first.weight[:, :128], first.bias)
        order_projection = F.linear(orders, first.weight[:, 128:], None)
        picks = last(torch.tanh(context_projection.unsqueeze(-2) + order_projection)).squeeze(-1)
        return torch.cat((picks, self.other_head(context)), -1), self.critic(context).squeeze(-1)


def masked_argmax(model, observations, action_masks, tie_tolerance=1e-5):
    """Action-only inference; no categorical distribution/log-probability work.

    This is separate from act/evaluate_actions used by on-policy training.
    """
    logits, _ = model(observations)
    if action_masks.dtype != torch.bool or action_masks.shape != logits.shape:
        raise ValueError('boolean action mask must match logits')
    if not action_masks.any(-1).all() or not torch.isfinite(logits).all():
        raise ValueError('nonempty legal set and finite logits required')
    masked = logits.masked_fill(~action_masks, -torch.inf)
    actions = masked.argmax(-1)
    if isinstance(model, FactoredStructuredInference):
        values = masked.topk(2, dim=-1).values
        close = values[..., 0] - values[..., 1] <= tie_tolerance
        if close.any():
            flat_obs = observations.reshape(-1, observations.shape[-1])
            flat_mask = action_masks.reshape(-1, action_masks.shape[-1])
            flat_close = close.reshape(-1)
            # Reference implementation explicitly, using the same weight tensors.
            original, _ = StructuredActorCritic.forward(model, flat_obs[flat_close])
            from medical_uav.agents.networks import masked_distribution
            historical = masked_distribution(original, flat_mask[flat_close]).probs.argmax(-1)
            actions = actions.clone()
            actions.reshape(-1)[flat_close] = historical
    return actions


class ActionOnlyEvaluationPolicy(torch.nn.Module):
    """Adapter for the existing evaluator, which consumes only the action.

    There are deliberately no log probabilities or values in this interface;
    collector/PPO continue to use the original full actor-critic model instead.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def act(self, observations, action_masks, deterministic=False):
        if not deterministic:
            raise ValueError('action-only policy is for deterministic evaluation only')
        return masked_argmax(self.model, observations, action_masks), None, None
