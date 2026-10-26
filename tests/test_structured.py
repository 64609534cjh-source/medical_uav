"""Structural invariants plus the real collector/update/checkpoint integration."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from medical_uav.agents.networks import (MLPActorCritic, StructuredActorCritic,
                                         build_network, masked_distribution)
from medical_uav.baselines.heuristics import slack_aware_policy
from medical_uav.env import MedicalUAVEnv
from medical_uav.training.trainer import TrainConfig, load_checkpoint, train


def real_state():
    env = MedicalUAVEnv()
    obs, info = env.reset(options={"demand_seed": 100003, "load": "high"})
    for _ in range(100):
        pending = obs[31:355].reshape(12, 27)
        cargo = obs[355:].reshape(3, 16)
        if pending[:, 0].sum() >= 2 and cargo[:, 0].sum() >= 1:
            break
        obs, _, done, _, info = env.step(slack_aware_policy(env))
        assert not done
    env.close()
    assert pending[:, 0].sum() >= 2 and cargo[:, 0].sum() >= 1
    return torch.from_numpy(obs), torch.from_numpy(info["action_mask"])


def test_equivariant_pending_invariant_cargo_and_masked_probabilities():
    torch.manual_seed(8)
    model = StructuredActorCritic()
    obs, mask = real_state()
    perm, cargo_perm = torch.randperm(12), torch.tensor([2, 0, 1])
    changed = torch.cat((obs[:31], obs[31:355].reshape(12, 27)[perm].flatten(),
                         obs[355:].reshape(3, 16)[cargo_perm].flatten()))
    perm_mask = torch.cat((mask[:12][perm], mask[12:]))
    logits, value = model(obs)
    other, other_value = model(changed)
    expected = torch.cat((logits[:12][perm], logits[12:]))
    torch.testing.assert_close(other, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(value, other_value)
    probs = masked_distribution(logits, mask).probs
    torch.testing.assert_close(masked_distribution(other, perm_mask).probs,
                               torch.cat((probs[:12][perm], probs[12:])))
    assert sum(p.numel() for p in model.parameters()) == 65_232
    for batch in (obs, obs.repeat(4, 1), obs.repeat(2, 4, 1)):
        logits, values = model(batch)
        assert logits.shape == (*batch.shape[:-1], 26)
        assert values.shape == batch.shape[:-1]


def test_empty_slots_mask_encoder_bias_and_have_no_input_gradient():
    torch.manual_seed(4)
    model = StructuredActorCritic()
    for size, encoder in ((27, model.pending_encoder), (16, model.cargo_encoder)):
        rows = torch.zeros(2, 5, size, requires_grad=True)
        embeddings, pool = model.encode_set(rows, encoder)
        assert torch.equal(embeddings, torch.zeros_like(embeddings))
        assert torch.equal(pool, torch.zeros_like(pool))
        pool.sum().backward()
        assert torch.equal(rows.grad, torch.zeros_like(rows.grad))
        noisy = torch.randn_like(rows.detach()); noisy[..., 0] = 0
        assert torch.equal(model.encode_set(noisy, encoder)[1], pool.detach())


def test_present_but_illegal_orders_still_contribute_to_state():
    torch.manual_seed(5)
    model = StructuredActorCritic()
    obs, mask = real_state()
    # Explicit action illegality never removes a present row from the encoder.
    present = int(torch.where(obs[31:355].reshape(12, 27)[:, 0] > 0)[0][0])
    mask[present] = False
    before, value = model(obs)
    changed = obs.clone(); changed[31 + present * 27 + 17] += 1
    after, next_value = model(changed)
    assert not torch.allclose(value, next_value, atol=1e-9, rtol=1e-9)
    assert not torch.allclose(before[12:], after[12:], atol=1e-9, rtol=1e-9)
    dist = masked_distribution(after, mask)
    assert dist.probs[present] == 0
    samples = dist.sample((200,))
    assert mask[samples].all()


def test_real_ppo_update_reload_and_architecture_guard(tmp_path):
    config = TrainConfig(architecture="structured", total_steps=1024,
                         evaluate_initial=False, evaluate_baselines=False, eval_split="train")
    train(config, tmp_path / "structured")
    state = load_checkpoint(tmp_path / "structured/last.pt")
    model = build_network(state["architecture"])
    model.load_state_dict(state["model_state_dict"])
    obs, mask = real_state()
    a, logp, value = model.act(obs, mask, deterministic=True)
    old, entropy, other_value = model.evaluate_actions(obs, mask, a)
    torch.testing.assert_close(logp, old); torch.testing.assert_close(value, other_value)
    assert torch.isfinite(entropy)
    assert min(v["step"] for v in state["optimizer_state_dict"]["state"].values()) > 0
    train(replace(config, total_steps=2048), tmp_path / "resumed", tmp_path / "structured/last.pt")
    with pytest.raises(RuntimeError):
        MLPActorCritic().load_state_dict(state["model_state_dict"])
    state["architecture"] = MLPActorCritic.architecture
    torch.save(state, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="disagrees"):
        load_checkpoint(tmp_path / "bad.pt")


def test_historical_mlp_checkpoint_defaults_and_exact_output(tmp_path):
    torch.manual_seed(42)
    model = MLPActorCritic()
    config = TrainConfig(total_steps=1024)
    old_config = dict(config.__dict__); old_config.pop("architecture")
    from dataclasses import asdict
    old_config["ppo"] = asdict(config.ppo)
    torch.save({"format_version": 1, "architecture": model.architecture,
                "config": old_config, "model_state_dict": model.state_dict()}, tmp_path / "old.pt")
    state = load_checkpoint(tmp_path / "old.pt")
    copy = build_network(state["architecture"]); copy.load_state_dict(state["model_state_dict"])
    obs, mask = real_state()
    assert all(torch.equal(a, b) for a, b in zip(model(obs), copy(obs)))
    assert TrainConfig.from_dict(old_config).architecture == "mlp"
