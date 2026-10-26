"""GAE over [time, environment], with distinct bootstrap and recurrence boundaries."""

import torch


@torch.no_grad()
def generalized_advantage(rewards, values, next_values, terminated, truncated,
                          gamma=1.0, gae_lambda=0.95):
    """Return raw advantages and fixed value targets before Actor normalization.

    next_values must describe the final post-action observation BEFORE any reset.
    Truncation bootstraps that value but stops recurrence into the next episode.
    The end of this tensor stops recurrence while retaining bootstrap.
    """
    if rewards.ndim != 2 or any(x.shape != rewards.shape for x in
                              (values, next_values, terminated, truncated)):
        raise ValueError("all GAE fields must have the same [T,N] shape")
    if terminated.dtype != torch.bool or truncated.dtype != torch.bool:
        raise ValueError("termination and truncation fields must be boolean")
    if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("gamma and lambda must lie in [0,1]")
    if not all(torch.isfinite(x).all() for x in (rewards, values, next_values)):
        raise ValueError("non-finite GAE input")
    advantages = torch.empty_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (~terminated[t]).float() * next_values[t] - values[t]
        carry = delta + gamma * gae_lambda * (~(terminated[t] | truncated[t])).float() * carry
        advantages[t] = carry
    return advantages, advantages + values
