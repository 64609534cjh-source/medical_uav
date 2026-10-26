"""The action mask must affect sampling, log probability and differentiable entropy."""

import pytest
import torch

from medical_uav.agents.networks import MLPActorCritic, masked_distribution


def test_masked_distribution_has_legal_support_and_finite_gradients():
    logits = torch.tensor([[0.0, 100.0, 1.0], [1.0, 0.0, -1.0]], requires_grad=True)
    masks = torch.tensor([[True, False, True], [False, True, False]])
    distribution = masked_distribution(logits, masks)
    samples = distribution.sample((200,))
    assert (samples[:, 0] != 1).all()
    assert (samples[:, 1] == 1).all()
    assert torch.equal(distribution.probs[~masks], torch.zeros(3))
    loss = -distribution.log_prob(torch.tensor([2, 1])).mean() - distribution.entropy().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad[~masks] == 0).all()
    assert logits.grad[0, masks[0]].abs().sum() > 0


@pytest.mark.parametrize("mask", [torch.zeros(2, 3, dtype=torch.bool),
                                  torch.ones(2, 3),
                                  torch.ones(3, dtype=torch.bool)])
def test_invalid_masks_are_rejected(mask):
    with pytest.raises(ValueError):
        masked_distribution(torch.zeros(2, 3), mask)


def test_network_rejects_nonfinite_logits():
    with pytest.raises(ValueError):
        masked_distribution(torch.tensor([float("nan"), 0.0]), torch.ones(2, dtype=torch.bool))


def test_architecture_matches_environment_interface():
    from medical_uav.env.actions import N_ACTIONS
    from medical_uav.env.observation import OBS_DIM

    model = MLPActorCritic()
    logits, values = model(torch.zeros(8, OBS_DIM))
    assert logits.shape == (8, N_ACTIONS)
    assert values.shape == (8,)
    assert sum(p.numel() for p in model.parameters()) == 139_803
