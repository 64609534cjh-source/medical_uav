"""Optional action-only inference: shared context projections, unchanged weights.

Never registered as a training architecture. The wrapped original network owns
all parameters; near ties fall back to its canonical categorical argmax.
"""
import torch
from torch import nn
from torch.nn import functional as F
from medical_uav.agents.decision import (state_and_destinations, urgency_features,
                                         UrgentDestinationActorCritic, CoordinatedDestinationActorCritic)
from medical_uav.agents.relational import PENDING_DEST, DestinationActorCritic
from medical_uav.env.actions import N_SLOTS


def factored_forward(model, observations):
    if not isinstance(model, DestinationActorCritic):
        raise ValueError('destination architecture required')
    c, orders, h, pending, cargo = state_and_destinations(model, observations)
    pick_local = orders
    if isinstance(model, CoordinatedDestinationActorCritic):
        membership = pending[..., PENDING_DEST:PENDING_DEST + 6] * (pending[..., :1] > 0)
        pick_local = torch.cat((orders, torch.matmul(membership, h)), -1)
    first, _, last = model.pick_head
    projected = F.linear(c, first.weight[:, :128], first.bias).unsqueeze(-2)
    picks = last(torch.tanh(projected + F.linear(pick_local, first.weight[:, 128:]))).squeeze(-1)
    deliver_local = h
    if isinstance(model, UrgentDestinationActorCritic):
        deliver_local = torch.cat((h, urgency_features(cargo, model.type_limits)), -1)
    first, _, last = model.deliver_head
    projected = F.linear(c, first.weight[:, :128], first.bias).unsqueeze(-2)
    delivers = last(torch.tanh(projected + F.linear(deliver_local, first.weight[:, 128:]))).squeeze(-1)
    return torch.cat((picks, delivers, model.other_head(c)), -1), model.critic(c).squeeze(-1)


class DestinationInferencePolicy(nn.Module):
    def __init__(self, model):
        super().__init__();self.model=model;self.fallback_states=0

    def forward(self, observations):
        return factored_forward(self.model, observations)

    def act(self, observations, action_masks, deterministic=False):
        if not deterministic:raise ValueError('action-only inference requires deterministic evaluation')
        logits, _ = self(observations)
        if action_masks.dtype != torch.bool or action_masks.shape != logits.shape:
            raise ValueError('boolean action masks matching logits required')
        if not action_masks.any(-1).all() or not torch.isfinite(logits).all():
            raise ValueError('nonempty legal masks and finite logits required')
        masked=logits.masked_fill(~action_masks,-torch.inf);actions=masked.argmax(-1)
        top=masked.topk(2,-1).values;close=top[...,0]-top[...,1]<=1e-5
        if close.any():
            # Preserve the original batch shape when resolving numerical ties.
            canonical=self.model.act(observations,action_masks,True)[0]
            actions=torch.where(close,canonical,actions);self.fallback_states+=int(close.sum())
        return actions,None,None
