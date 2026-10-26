"""Generation 3: separate urgency and pickup/delivery coordination hypotheses.

Only current observation fields are read. The frozen Gen2 architecture is not
modified. Destination summaries include the visible candidate itself: these are
current-state summaries, never simulated post-pick quantities.
"""
import torch
from torch import nn
from medical_uav.agents.relational import (DestinationActorCritic, split_observation,
                                           destination_groups, PENDING_DEST, CARGO_DEST, TRAVEL_START)
from medical_uav.env.actions import N_SLOTS
from medical_uav.env.config import FACILITIES, N_ONBOARD, ORDER_TYPES

N_FACILITIES = len(FACILITIES)
URGENCY_NAMES = ('live_count', 'live_weight', 'has_live', 'min_deadline_minutes',
                 'min_expiry_minutes', 'live_due_within_15min_count', 'expired_count',
                 'expired_weight', 'has_other_live', 'other_min_deadline_minutes',
                 'other_min_expiry_minutes')


def state_and_destinations(model, observations):
    """Same feature definitions and concatenation order as frozen Gen2 DELIVER."""
    g, pending, cargo = split_observation(observations)
    orders, order_pool = model.encode_set(pending, model.pending_encoder)
    boxes, cargo_pool = model.encode_set(cargo, model.cargo_encoder)
    context = model.context_encoder(torch.cat((model.global_encoder(g), order_pool, cargo_pool), -1))
    pending_group, npending = destination_groups(pending, orders, PENDING_DEST)
    cargo_group, ncargo = destination_groups(cargo, boxes, CARGO_DEST)
    facility = model.facility_embedding.weight.expand(*context.shape[:-1], N_FACILITIES, 8)
    travel = g[..., TRAVEL_START:TRAVEL_START + N_FACILITIES]
    h = torch.cat((cargo_group, pending_group, (ncargo / N_ONBOARD).unsqueeze(-1),
                   (npending / N_SLOTS).unsqueeze(-1), travel.unsqueeze(-1), facility), -1)
    return context, orders, h, pending, cargo


def grouped_min(values, membership):
    """Exactly zero for empty groups; input values have physical minute units."""
    expanded = values.unsqueeze(-2).expand_as(membership)
    smallest = expanded.masked_fill(~membership, torch.inf).amin(-1)
    return torch.where(membership.any(-1), smallest, torch.zeros_like(smallest))


def urgency_features(cargo, type_limits):
    """Eleven cargo-risk statistics per facility, reconstructed from observation.

    Deadline/expiry entries are scaled by each order's TYPE LIMIT, not a common
    unit. Convert to minutes before cross-type minima or the fixed 15min test.
    Original observation clipping cannot be undone; no simulator values are read.
    """
    present = cargo[..., 0] > 0
    expired = present & (cargo[..., 14] > .5)
    live = present & ~expired
    dest = cargo[..., CARGO_DEST:CARGO_DEST + N_FACILITIES].transpose(-1, -2) > .5
    group = dest & live.unsqueeze(-2)
    expired_group = dest & expired.unsqueeze(-2)
    other = ~dest & live.unsqueeze(-2)
    limits = (cargo[..., 1:5] * type_limits).sum(-1)
    deadline = cargo[..., 12] * limits
    expiry = cargo[..., 13] * limits
    weight = cargo[..., 11].unsqueeze(-2)
    time_scale = lambda x: x.clamp(-300, 300) / 30.0
    return torch.stack((group.sum(-1) / 3.0, (group * weight).sum(-1) / 2.0,
                        group.any(-1).to(cargo.dtype), time_scale(grouped_min(deadline, group)),
                        time_scale(grouped_min(expiry, group)),
                        (group & (deadline <= 15.0).unsqueeze(-2)).sum(-1) / 3.0,
                        expired_group.sum(-1) / 3.0, (expired_group * weight).sum(-1) / 2.0,
                        other.any(-1).to(cargo.dtype), time_scale(grouped_min(deadline, other)),
                        time_scale(grouped_min(expiry, other))), -1)


class UrgentDestinationActorCritic(DestinationActorCritic):
    """Only augment DELIVER with explicit cargo urgency/capacity risk statistics."""
    architecture = 'urgent_destination_403_26_v1'

    def __init__(self):
        super().__init__()
        self.register_buffer('type_limits', torch.tensor([t.limit for t in ORDER_TYPES]), persistent=False)
        self.deliver_head = nn.Sequential(nn.Linear(246, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, observations):
        context, orders, h, _, cargo = state_and_destinations(self, observations)
        c = context.unsqueeze(-2)
        picks = self.pick_head(torch.cat((c.expand(*context.shape[:-1], N_SLOTS, 128), orders), -1)).squeeze(-1)
        features = torch.cat((c.expand(*context.shape[:-1], N_FACILITIES, 128), h,
                              urgency_features(cargo, self.type_limits)), -1)
        delivers = self.deliver_head(features).squeeze(-1)
        return torch.cat((picks, delivers, self.other_head(context)), -1), self.critic(context).squeeze(-1)


class CoordinatedDestinationActorCritic(DestinationActorCritic):
    """Only PICK gains the same 107-dimensional destination context as DELIVER."""
    architecture = 'coordinated_destination_403_26_v1'

    def __init__(self):
        super().__init__()
        self.pick_head = nn.Sequential(nn.Linear(299, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, observations):
        context, orders, h, pending, _ = state_and_destinations(self, observations)
        membership = pending[..., PENDING_DEST:PENDING_DEST + N_FACILITIES] * (pending[..., :1] > 0)
        order_destination = torch.matmul(membership, h)
        c = context.unsqueeze(-2)
        picks = self.pick_head(torch.cat((c.expand(*context.shape[:-1], N_SLOTS, 128), orders,
                                         order_destination), -1)).squeeze(-1)
        delivers = self.deliver_head(torch.cat((c.expand(*context.shape[:-1], N_FACILITIES, 128), h), -1)).squeeze(-1)
        return torch.cat((picks, delivers, self.other_head(context)), -1), self.critic(context).squeeze(-1)
