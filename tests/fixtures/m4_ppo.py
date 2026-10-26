"""Clipped PPO updates on a fixed rollout with historical legal-action masks.

The value loss is MSE without an additional 1/2 factor. Gamma=1 keeps the objective
as undiscounted full-day reward; elapsed time is logged, not used as a discount.
"""

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.nn import functional as F


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    epochs: int = 4
    minibatch_size: int = 256
    clip_coef: float = 0.2
    gamma: float = 1.0
    gae_lambda: float = 0.95
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.02

    def validate(self):
        if not all(np.isfinite(x) for x in asdict(self).values()):
            raise ValueError("PPO parameters must be finite")
        if self.epochs < 1 or self.minibatch_size < 1 or self.learning_rate <= 0:
            raise ValueError("epochs, minibatch size and learning rate must be positive")
        if not 0 < self.clip_coef < 1 or not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("invalid clipping, discount or GAE coefficient")
        if min(self.value_coef, self.entropy_coef, self.target_kl) < 0 or self.max_grad_norm <= 0:
            raise ValueError("invalid loss, KL or gradient coefficient")


def clipped_policy_loss(new_log_prob, old_log_prob, advantages, clip_coef):
    log_ratio = new_log_prob - old_log_prob
    ratio = log_ratio.exp()
    loss = torch.maximum(-advantages * ratio,
                         -advantages * ratio.clamp(1 - clip_coef, 1 + clip_coef)).mean()
    return loss, ratio, log_ratio


class PPOUpdater:
    def __init__(self, network, config=None, seed=0):
        self.network, self.config = network, config or PPOConfig()
        self.config.validate()
        self.optimizer = torch.optim.Adam(network.parameters(), lr=self.config.learning_rate)
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))

    def update(self, buffer):
        cfg, batch = self.config, buffer.flattened()
        size = len(batch["action"])
        with torch.no_grad():
            old_check, _, _ = self.network.evaluate_actions(
                batch["obs"], batch["action_mask"], batch["action"])
            initial_ratio_error = float(((old_check - batch["old_log_prob"]).exp() - 1).abs().max())
        if initial_ratio_error > 1e-5:
            raise RuntimeError(f"old-policy ratio differs from one: {initial_ratio_error}")
        raw_adv = batch["advantage"]
        actor_adv = (raw_adv - raw_adv.mean()) / (raw_adv.std(unbiased=False) + 1e-8)
        # The raw-advantage returns above stay unchanged throughout every epoch.
        records, stopped, completed_epochs = [], False, 0
        for _ in range(cfg.epochs):
            for indices in np.array_split(self.rng.permutation(size),
                                          int(np.ceil(size / cfg.minibatch_size))):
                log_prob, entropy, values = self.network.evaluate_actions(
                    batch["obs"][indices], batch["action_mask"][indices], batch["action"][indices])
                policy_loss, ratio, log_ratio = clipped_policy_loss(
                    log_prob, batch["old_log_prob"][indices], actor_adv[indices], cfg.clip_coef)
                approx_kl = ((ratio - 1) - log_ratio).mean()
                if not torch.isfinite(approx_kl):
                    raise RuntimeError("non-finite PPO probability ratio")
                if cfg.target_kl and float(approx_kl.detach()) > cfg.target_kl:
                    stopped = True
                    break
                value_loss = F.mse_loss(values, batch["returns"][indices])
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy.mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite PPO loss")
                self.optimizer.zero_grad()
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), cfg.max_grad_norm, error_if_nonfinite=True)
                self.optimizer.step()
                records.append((len(indices), {
                    "policy_loss": float(policy_loss.detach()), "value_loss": float(value_loss.detach()),
                    "entropy": float(entropy.mean().detach()), "approx_kl": float(approx_kl.detach()),
                    "clip_fraction": float(((ratio - 1).abs() > cfg.clip_coef).float().mean()),
                    "gradient_norm": float(gradient_norm),
                }))
            if stopped:
                break
            completed_epochs += 1
        with torch.no_grad():
            final_log_prob, _, final_values = self.network.evaluate_actions(
                batch["obs"], batch["action_mask"], batch["action"])
            final_log_ratio = final_log_prob - batch["old_log_prob"]
            final_kl = float(((final_log_ratio.exp() - 1) - final_log_ratio).mean())
            variance = batch["returns"].var(unbiased=False)
            explained_variance = (float(1 - (batch["returns"] - final_values).var(unbiased=False) / variance)
                                  if variance > 1e-12 else None)
        if not records or not all(torch.isfinite(p).all() for p in self.network.parameters()):
            raise RuntimeError("PPO update did not produce finite updated parameters")
        weight = sum(n for n, _ in records)
        result = {k: sum(n * r[k] for n, r in records) / weight for k in records[0][1]}
        result.update({
            "initial_ratio_max_error": initial_ratio_error, "final_approx_kl": final_kl,
            "advantage_mean": float(raw_adv.mean()), "advantage_std": float(raw_adv.std(unbiased=False)),
            "return_mean": float(batch["returns"].mean()),
            "return_std": float(batch["returns"].std(unbiased=False)),
            "explained_variance": explained_variance, "optimizer_steps": len(records),
            "epochs_completed": completed_epochs, "kl_early_stop": stopped,
        })
        return result
