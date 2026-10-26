"""Demonstrate the SAME masked PPO components used by the continuous trainer.

One rollout and one PPO cycle verify forward/backward and checkpoint behavior.
This is not evidence of learned performance. Only Train demand days are used.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from medical_uav.agents.networks import MLPActorCritic, masked_distribution
from medical_uav.algorithms.ppo import PPOConfig, PPOUpdater
from medical_uav.training.collector import Collector


def run_demo(output_dir, batch_size=256, seed=0, threads=1):
    if batch_size < 1 or threads < 1 or not 0 <= seed < 100_000:
        raise ValueError("positive batch/threads and a Train seed (0..99999) are required")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    started = time.perf_counter()
    network = MLPActorCritic()
    updater = PPOUpdater(network, PPOConfig(), seed)
    collector = Collector(num_envs=1, seed=seed, loads=("medium",))
    tick = time.perf_counter()
    try:
        buffer, episodes = collector.collect(network, batch_size)
    finally:
        collector.close()
    collection_seconds = time.perf_counter() - tick
    buffer.finish(updater.config.gamma, updater.config.gae_lambda)
    batch = buffer.flattened()
    obs_batch, mask_batch = batch["obs"], batch["action_mask"]
    with torch.no_grad():
        logits, values = network(obs_batch)
        distribution = masked_distribution(logits, mask_batch)
    before = {name: param.detach().clone() for name, param in network.named_parameters()}
    tick = time.perf_counter()
    diagnostics = updater.update(buffer)
    update_seconds = time.perf_counter() - tick
    actor_grad_norm = float(network.actor.weight.grad.norm())
    critic_grad_norm = float(network.critic.weight.grad.norm())
    changed = [name for name, param in network.named_parameters() if not torch.equal(before[name], param)]
    if "actor.weight" not in changed or "critic.weight" not in changed:
        raise RuntimeError("both heads must change after PPO")

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
        "purpose": "one real masked PPO cycle; not a learning benchmark",
        "ppo": diagnostics,
        "python": platform.python_version(), "torch": torch.__version__,
        "device": "cpu", "cuda_available": torch.cuda.is_available(),
        "threads": threads, "seed": seed,
        "parameters": sum(p.numel() for p in network.parameters()),
        "transitions": batch_size, "completed_days": len(episodes),
        "observation_shape": list(obs_batch.shape), "logits_shape": list(logits.shape),
        "value_shape": list(values.shape), "invalid_actions": 0,
        "example_before_update": {
            "observation_first_eight": obs_batch[0, :8].tolist(),
            "legal_action_indices": mask_batch[0].nonzero().flatten().tolist(),
            "masked_action_probabilities": distribution.probs[0].detach().tolist(),
            "sampled_action": int(batch["action"][0]), "value": float(values[0].detach()),
            "reward": float(buffer.reward[0, 0]),
        },
        "multi_action_states": int((mask_batch.sum(-1) > 1).sum()),
        "loss": diagnostics["policy_loss"] + 0.5 * diagnostics["value_loss"] - 0.01 * diagnostics["entropy"], "actor_gradient_norm": actor_grad_norm,
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
