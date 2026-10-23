"""Policy/value networks. Each maps a batch of observations and action masks to
masked action logits [B, 26] and state values [B]."""
import torch

from medical_uav.env.actions import N_SLOTS
from medical_uav.env.config import N_ONBOARD
from medical_uav.env.observation import GLOBAL_DIM, ONBOARD_DIM, PENDING_DIM

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
