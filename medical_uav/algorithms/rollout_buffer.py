"""One immutable old-policy rollout after collection; flatten only after GAE."""

import torch

from medical_uav.algorithms.returns import generalized_advantage
from medical_uav.env.actions import N_ACTIONS
from medical_uav.env.observation import OBS_DIM


class RolloutBuffer:
    def __init__(self, steps, num_envs):
        if steps < 1 or num_envs < 1:
            raise ValueError("positive rollout and environment counts are required")
        self.steps, self.num_envs = steps, num_envs
        shape = (steps, num_envs)
        self.obs = torch.empty((*shape, OBS_DIM), dtype=torch.float32)
        self.action_mask = torch.empty((*shape, N_ACTIONS), dtype=torch.bool)
        self.action = torch.empty(shape, dtype=torch.int64)
        for name in ("old_log_prob", "old_value", "reward", "elapsed", "next_value"):
            setattr(self, name, torch.empty(shape, dtype=torch.float32))
        self.terminated = torch.empty(shape, dtype=torch.bool)
        self.truncated = torch.empty(shape, dtype=torch.bool)
        self.advantage = self.returns = None

    def finish(self, gamma=1.0, gae_lambda=0.95):
        self.advantage, self.returns = generalized_advantage(
            self.reward, self.old_value, self.next_value, self.terminated, self.truncated,
            gamma, gae_lambda,
        )

    def flattened(self):
        if self.advantage is None:
            raise RuntimeError("finish GAE on the time axis before flattening")
        return {name: getattr(self, name).flatten(0, 1) for name in (
            "obs", "action_mask", "action", "old_log_prob", "old_value",
            "advantage", "returns",
        )}
