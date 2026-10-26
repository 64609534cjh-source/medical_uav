"""Generation 2: two independent, observation-only changes to the M5 network.

No simulator access, slot-position embeddings, new actions, or oracle features.
The original StructuredActorCritic implementation/state dictionaries stay intact.
"""

import torch
from torch import nn

from medical_uav.agents.networks import StructuredActorCritic
from medical_uav.env.actions import N_SLOTS
from medical_uav.env.config import FACILITIES, N_NODES, N_ONBOARD
from medical_uav.env.observation import GLOBAL_DIM, PENDING_DIM, ONBOARD_DIM, N_TYPES

PENDING_END = GLOBAL_DIM + N_SLOTS * PENDING_DIM
PENDING_DEST = 1 + N_TYPES + len(FACILITIES)
CARGO_DEST = 1 + N_TYPES
TRAVEL_START = 9 + N_NODES


def split_observation(observations):
    leading = observations.shape[:-1]
    return (observations[..., :GLOBAL_DIM],
            observations[..., GLOBAL_DIM:PENDING_END].reshape(*leading, N_SLOTS, PENDING_DIM),
            observations[..., PENDING_END:].reshape(*leading, N_ONBOARD, ONBOARD_DIM))


def destination_groups(rows, embeddings, dest_start):
    """Mean embeddings and raw counts by destination; physical expired cargo counts.

    Presence, not PICK legality or expiry, determines membership. Absent groups
    are exactly zero even when the shared feature encoders have nonzero biases.
    """
    membership = rows[..., dest_start:dest_start + len(FACILITIES)] * (rows[..., :1] > 0)
    counts = membership.sum(-2)
    pooled = torch.matmul(membership.transpose(-1, -2), embeddings) / counts.clamp_min(1).unsqueeze(-1)
    return pooled, counts


class DestinationActorCritic(StructuredActorCritic):
    """M5 shared features/PICK/critic; only six DELIVER outputs are replaced."""

    architecture = 'destination_mean_403_26_v1'

    def __init__(self):
        super().__init__()
        # Remove the six replaced outputs rather than retain unused parameters.
        self.other_head = nn.Linear(128, 8)  # six CHARGE, RETURN, WAIT
        self.facility_embedding = nn.Embedding(len(FACILITIES), 8)
        self.deliver_head = nn.Sequential(nn.Linear(235, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, observations):
        g, pending, cargo = split_observation(observations)
        orders, order_pool = self.encode_set(pending, self.pending_encoder)
        boxes, cargo_pool = self.encode_set(cargo, self.cargo_encoder)
        context = self.context_encoder(torch.cat((self.global_encoder(g), order_pool, cargo_pool), -1))
        pick_context = context.unsqueeze(-2).expand(*context.shape[:-1], N_SLOTS, 128)
        picks = self.pick_head(torch.cat((pick_context, orders), -1)).squeeze(-1)
        pending_group, pending_count = destination_groups(pending, orders, PENDING_DEST)
        cargo_group, cargo_count = destination_groups(cargo, boxes, CARGO_DEST)
        # Fixed normalizers; all quantities derive from the existing 403 features.
        travel = g[..., TRAVEL_START:TRAVEL_START + len(FACILITIES)]
        facility = self.facility_embedding.weight.expand(*context.shape[:-1], len(FACILITIES), 8)
        facility_context = context.unsqueeze(-2).expand(*context.shape[:-1], len(FACILITIES), 128)
        features = torch.cat((facility_context, cargo_group, pending_group,
                              (cargo_count / N_ONBOARD).unsqueeze(-1),
                              (pending_count / N_SLOTS).unsqueeze(-1), travel.unsqueeze(-1), facility), -1)
        delivers = self.deliver_head(features).squeeze(-1)
        return torch.cat((picks, delivers, self.other_head(context)), -1), self.critic(context).squeeze(-1)


class RelationalActorCritic(StructuredActorCritic):
    """One pre-norm self-attention block on 16 tokens, no positional encodings.

    The cargo projections preserve the M5 160-dimensional context interface.
    Dropout is zero, including in training. Explicit per-head attention avoids
    train/eval fast-path changes and makes padding semantics inspectable.
    """

    architecture = 'relational_16x64_h4_ff128_403_26_v1'

    def __init__(self):
        super().__init__()
        self.cargo_to_token = nn.Linear(32, 64)
        self.token_to_cargo = nn.Linear(64, 32)
        self.norm_attention = nn.LayerNorm(64)
        self.qkv = nn.Linear(64, 192)
        self.attention_output = nn.Linear(64, 64)
        self.norm_ff = nn.LayerNorm(64)
        self.ff = nn.Sequential(nn.Linear(64, 128), nn.Tanh(), nn.Linear(128, 64))

    def forward(self, observations):
        leading = observations.shape[:-1]
        g, pending, cargo = split_observation(observations)
        orders, _ = self.encode_set(pending, self.pending_encoder)
        boxes, _ = self.encode_set(cargo, self.cargo_encoder)
        tokens = torch.cat((self.global_encoder(g).unsqueeze(-2), orders, self.cargo_to_token(boxes)), -2)
        present = torch.cat((torch.ones_like(g[..., :1], dtype=torch.bool), pending[..., 0] > 0, cargo[..., 0] > 0), -1)
        x = tokens.reshape(-1, 16, 64)
        present_flat = present.reshape(-1, 16)
        qkv = self.qkv(self.norm_attention(x)).reshape(-1, 16, 3, 4, 16)
        q, k, v = [qkv[:, :, i].transpose(1, 2) for i in range(3)]
        scores = torch.matmul(q, k.transpose(-1, -2)) / 4.0
        scores = scores.masked_fill(~present_flat[:, None, None, :], -torch.inf)
        attended = torch.matmul(scores.softmax(-1), v).transpose(1, 2).reshape(-1, 16, 64)
        x = x + self.attention_output(attended)
        x = x + self.ff(self.norm_ff(x))
        x = x.reshape(*leading, 16, 64) * present.unsqueeze(-1)
        updated_orders = x[..., 1:13, :]
        order_pool = updated_orders.sum(-2) / present[..., 1:13].sum(-1).clamp_min(1).unsqueeze(-1)
        updated_cargo = self.token_to_cargo(x[..., 13:, :]) * present[..., 13:].unsqueeze(-1)
        cargo_pool = updated_cargo.sum(-2) / present[..., 13:].sum(-1).clamp_min(1).unsqueeze(-1)
        context = self.context_encoder(torch.cat((x[..., 0, :], order_pool, cargo_pool), -1))
        expanded = context.unsqueeze(-2).expand(*leading, N_SLOTS, 128)
        picks = self.pick_head(torch.cat((expanded, updated_orders), -1)).squeeze(-1)
        return torch.cat((picks, self.other_head(context)), -1), self.critic(context).squeeze(-1)
