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


def _relu_mlp(sizes: list[int], last_relu: bool) -> torch.nn.Sequential:
    """Linear layers with ReLU between them, and after the last one if last_relu."""
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_relu:
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


class SetNet(torch.nn.Module):
    """Set/attention network from the proposal: orders and cargo are encoded one slot at a
    time by shared encoders, the UAV state attends over them, and every PICK slot is scored
    by the same scorer, so reordering the slots reorders the PICK logits."""

    def __init__(self, width: int = 64, heads: int = 4):
        super().__init__()
        self.pending_enc = _relu_mlp([PENDING_DIM, width, width], last_relu=True)
        self.cargo_enc = _relu_mlp([ONBOARD_DIM, width], last_relu=True)
        self.global_enc = _relu_mlp([GLOBAL_DIM, width], last_relu=True)
        self.null = torch.nn.Parameter(0.02 * torch.randn(1, 1, width))  # always-valid key
        self.attn = torch.nn.MultiheadAttention(width, heads, batch_first=True)
        self.pick_head = _relu_mlp([3 * width, width, 1], last_relu=False)
        self.other_head = _relu_mlp([2 * width, width, N_ACTIONS - N_SLOTS], last_relu=False)
        self.value_head = _relu_mlp([3 * width, width, 1], last_relu=False)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g, pending, cargo = split_obs(obs)
        b = obs.shape[0]
        ge = self.global_enc(g)  # [B, 64]
        pe = self.pending_enc(pending)  # [B, 12, 64]
        ce = self.cargo_enc(cargo)  # [B, 3, 64]

        # the UAV state queries the null token, the orders and the cargo; empty slots are ignored
        keys = torch.cat([self.null.expand(b, -1, -1), pe, ce], dim=1)
        empty = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=obs.device),
                           pending[:, :, 0] == 0, cargo[:, :, 0] == 0], dim=1)
        attended, _ = self.attn(ge[:, None], keys, keys, key_padding_mask=empty)
        context = torch.cat([ge, attended[:, 0]], dim=1)  # [B, 128]

        # one shared scorer for every PICK slot, a small head for the 14 other actions
        pick = self.pick_head(torch.cat([pe, context[:, None].expand(-1, N_SLOTS, -1)], dim=2))
        logits = torch.cat([pick[:, :, 0], self.other_head(context)], dim=1)

        # value from the context and the mean of the occupied order slots (zero if none)
        occupied = (pending[:, :, 0] > 0).float()[:, :, None]
        mean = (pe * occupied).sum(1) / occupied.sum(1).clamp(min=1.0)
        value = self.value_head(torch.cat([context, mean], dim=1))[:, 0]
        return masked_logits(logits, mask), value


NETWORKS = {"mlp": MLPNet, "set": SetNet}  # network kinds by name, as stored in checkpoints
