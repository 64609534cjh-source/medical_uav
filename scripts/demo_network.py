"""Verify the CPU environment using real transitions and one actor-critic update.

This is an installation/forward/backward/checkpoint demonstration, not PPO training
or evidence of learned performance. No Dev or Test demand days are used.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from medical_uav.agents.networks import MLPActorCritic, masked_distribution
from medical_uav.env import MedicalUAVEnv


def run_demo(output_dir, batch_size=256, seed=0, threads=1):
    if batch_size < 1 or threads < 1 or not 0 <= seed < 100_000:
        raise ValueError("positive batch/threads and a Train seed (0..99999) are required")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    started = time.perf_counter()
    network = MLPActorCritic()
    optimizer = torch.optim.Adam(network.parameters(), lr=3e-4)
    observations, masks, actions, rewards, next_observations, terminals = [], [], [], [], [], []
    invalid_actions, completed_days = 0, 0
    env = MedicalUAVEnv()
    obs, info = env.reset(options={"demand_seed": seed, "load": "medium"})
    with torch.no_grad():
        for _ in range(batch_size):
            mask = info["action_mask"].copy()
            logits, _ = network(torch.from_numpy(obs))
            action = int(masked_distribution(logits, torch.from_numpy(mask)).sample())
            next_obs, reward, terminated, truncated, info = env.step(action)
            invalid_actions += int(info["invalid_action"])
            if invalid_actions:
                raise RuntimeError("masked policy selected an illegal action")
            observations.append(obs)
            masks.append(mask)
            actions.append(action)
            rewards.append(reward)
            next_observations.append(next_obs)
            terminals.append(terminated)
            obs = next_obs
            if terminated or truncated:
                completed_days += 1
                obs, info = env.reset(options={
                    "demand_seed": (seed + completed_days) % 100_000,
                    "load": ("medium", "high", "low")[completed_days % 3],
                })
    env.close()
    collection_seconds = time.perf_counter() - started

    obs_batch = torch.from_numpy(np.stack(observations))
    mask_batch = torch.from_numpy(np.stack(masks))
    action_batch = torch.tensor(actions)
    logits, values = network(obs_batch)
    distribution = masked_distribution(logits, mask_batch)
    # A one-step undiscounted TD target is sufficient to exercise both heads.
    # This is deliberately not a PPO objective or a training performance result.
    with torch.no_grad():
        _, next_values = network(torch.from_numpy(np.stack(next_observations)))
        targets = torch.tensor(rewards) + (~torch.tensor(terminals)).float() * next_values
    advantages = targets - values.detach()
    actor_loss = -(distribution.log_prob(action_batch) * advantages).mean()
    value_loss = F.mse_loss(values, targets)
    loss = actor_loss + 0.5 * value_loss - 0.01 * distribution.entropy().mean()
    before = {name: param.detach().clone() for name, param in network.named_parameters()}
    update_start = time.perf_counter()
    optimizer.zero_grad()
    loss.backward()
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in network.parameters()):
        raise RuntimeError("missing or non-finite gradients")
    actor_grad_norm = float(network.actor.weight.grad.norm())
    critic_grad_norm = float(network.critic.weight.grad.norm())
    if actor_grad_norm == 0 or critic_grad_norm == 0:
        raise RuntimeError("both heads must receive a nonzero gradient")
    torch.nn.utils.clip_grad_norm_(network.parameters(), 0.5)
    optimizer.step()
    update_seconds = time.perf_counter() - update_start
    changed = [name for name, param in network.named_parameters() if not torch.equal(before[name], param)]
    if "actor.weight" not in changed or "critic.weight" not in changed:
        raise RuntimeError("both heads must change after the optimizer step")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "network_smoke.pt"
    torch.save({"model_state_dict": network.state_dict(), "seed": seed}, checkpoint)
    restored = MLPActorCritic()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True)["model_state_dict"])
    with torch.no_grad():
        expected = network(obs_batch)
        actual = restored(obs_batch)
        if not all(torch.equal(a, b) for a, b in zip(expected, actual)):
            raise RuntimeError("checkpoint outputs changed after reload")
    result = {
        "purpose": "environment smoke test; not PPO training",
        "python": platform.python_version(), "torch": torch.__version__,
        "device": "cpu", "cuda_available": torch.cuda.is_available(),
        "threads": threads, "seed": seed,
        "parameters": sum(p.numel() for p in network.parameters()),
        "transitions": batch_size, "completed_days": completed_days,
        "observation_shape": list(obs_batch.shape), "logits_shape": list(logits.shape),
        "value_shape": list(values.shape), "invalid_actions": invalid_actions,
        "example_before_update": {
            "observation_first_eight": obs_batch[0, :8].tolist(),
            "legal_action_indices": mask_batch[0].nonzero().flatten().tolist(),
            "masked_action_probabilities": distribution.probs[0].detach().tolist(),
            "sampled_action": actions[0], "value": float(values[0].detach()),
            "reward": rewards[0],
        },
        "multi_action_states": int((mask_batch.sum(-1) > 1).sum()),
        "loss": float(loss.detach()), "actor_gradient_norm": actor_grad_norm,
        "critic_gradient_norm": critic_grad_norm,
        "changed_parameter_tensors": len(changed), "checkpoint_reload_exact": True,
        "collection_seconds": collection_seconds, "update_seconds": update_seconds,
        "total_seconds": time.perf_counter() - started,
    }
    (output_dir / "network_smoke.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/network_smoke"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(run_demo(args.output_dir, args.batch_size, args.seed, args.threads), indent=2))
