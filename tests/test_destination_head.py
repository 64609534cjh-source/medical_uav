import numpy as np
import pytest
import torch

from medical_uav.agents.networks import (NETWORKS, SetDestNet, SetNet, destination_means,
                                         split_obs)
from medical_uav.agents.policy import NetworkPolicy, load_checkpoint, save_checkpoint
from medical_uav.agents.ppo import RolloutBuffer, compute_gae, ppo_update
from medical_uav.env.actions import CHARGE0, DELIVER0, N_ACTIONS, N_SLOTS
from medical_uav.env.config import ENV_VERSION, FACILITIES, N_NODES, N_ONBOARD
from medical_uav.env.observation import GLOBAL_DIM, N_TYPES, OBS_DIM


def synthetic_batch(n=4):
    """Slot fixtures without generating demand or running a policy."""
    obs = torch.zeros(n, OBS_DIM)
    g, pending, cargo = split_obs(obs)
    g[:] = torch.rand_like(g)
    for slots, start, destinations in ((pending, 1 + N_TYPES + len(FACILITIES), [2, 2, 4]),
                                       (cargo, 1 + N_TYPES, [1, 2])):
        slots[..., 1:] = torch.rand_like(slots[..., 1:])
        slots[..., start:start + len(FACILITIES)] = 0
        for i, dest in enumerate(destinations):
            slots[:, i, 0] = 1
            slots[:, i, start + dest] = 1
    mask = torch.ones(n, N_ACTIONS, dtype=torch.bool)
    mask[:, [0, 3, CHARGE0]] = False
    return obs, mask


def test_destination_shapes_mask_and_batch_of_one():
    net = NETWORKS["set_dest"]()
    assert isinstance(net, SetDestNet)
    obs, mask = synthetic_batch()
    for o, m in ((obs, mask), (obs[:1], mask[:1])):
        logits, value = net(o, m)
        assert logits.shape == m.shape and value.shape == (len(o),)
        assert torch.softmax(logits, 1)[~m].abs().max() == 0
        assert torch.isfinite(logits).all() and torch.isfinite(value).all()


@pytest.mark.parametrize("pending_order,cargo_order", [(True, False), (False, True), (True, True)])
def test_destination_slot_permutations(pending_order, cargo_order):
    torch.manual_seed(0)
    net = SetDestNet()
    obs, mask = synthetic_batch()
    g, pending, cargo = split_obs(obs)
    pp = torch.randperm(N_SLOTS) if pending_order else torch.arange(N_SLOTS)
    cp = torch.randperm(N_ONBOARD) if cargo_order else torch.arange(N_ONBOARD)
    permuted = torch.cat([g, pending[:, pp].flatten(1), cargo[:, cp].flatten(1)], 1)
    mask_p = mask.clone()
    mask_p[:, :DELIVER0] = mask[:, :DELIVER0][:, pp]
    with torch.no_grad():
        logits, value = net(obs, mask)
        logits_p, value_p = net(permuted, mask_p)
    assert torch.allclose(logits_p[:, :DELIVER0], logits[:, :DELIVER0][:, pp], atol=1e-6)
    assert torch.allclose(logits_p[:, DELIVER0:], logits[:, DELIVER0:], atol=1e-6)
    assert torch.allclose(value_p, value, atol=1e-6)


@pytest.mark.parametrize("kind", ["pending", "cargo"])
def test_destination_means_ignore_empty_slots(kind):
    obs, _ = synthetic_batch(1)
    _, pending, cargo = split_obs(obs)
    slots = pending if kind == "pending" else cargo
    start = 1 + N_TYPES + (len(FACILITIES) if kind == "pending" else 0)
    emb = torch.arange(slots.shape[1] * 4, dtype=torch.float32).reshape(1, -1, 4) + 1
    slots[:, -1, start] = 1  # stale data in an unoccupied slot must not count
    means, counts = destination_means(slots, emb, start)
    empty = [0, 1, 3, 5] if kind == "pending" else [0, 3, 4, 5]
    assert torch.equal(means[:, empty], torch.zeros_like(means[:, empty]))
    assert torch.equal(counts[:, empty], torch.zeros_like(counts[:, empty]))
    if kind == "pending":
        assert torch.equal(means[:, 2], emb[:, :2].mean(1))
        assert torch.equal(means[:, 4], emb[:, 2])
        assert counts.tolist() == [[0, 0, 2, 0, 1, 0]]
    else:
        assert torch.equal(means[:, 1], emb[:, 0])
        assert torch.equal(means[:, 2], emb[:, 1])
        assert counts.tolist() == [[0, 1, 1, 0, 0, 0]]


def test_destination_features_use_observed_groups_counts_and_travel():
    torch.manual_seed(0)
    net = SetDestNet()
    obs, mask = synthetic_batch(1)
    g, pending, cargo = split_obs(obs)
    captured = []
    hook = net.deliver_head.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    with torch.no_grad():
        net(obs, mask)
    hook.remove()
    features = captured[0]
    cm, cc = destination_means(cargo, net.cargo_enc(cargo), 1 + N_TYPES)
    pm, pc = destination_means(pending, net.pending_enc(pending), 1 + N_TYPES + len(FACILITIES))
    assert features.shape == (1, 6, 267)
    assert torch.equal(features[:, :, 128:192], cm)
    assert torch.equal(features[:, :, 192:256], pm)
    assert torch.equal(features[:, :, 256], cc / N_ONBOARD)
    assert torch.equal(features[:, :, 257], pc / N_SLOTS)
    travel = g[:, 9 + N_NODES:9 + 2 * N_NODES][:, list(FACILITIES)]
    assert torch.equal(features[:, :, 258], travel)
    assert torch.equal(features[:, :, 259:], net.facility_emb.weight[None])


def test_destination_empty_and_single_groups_finite():
    net = SetDestNet()
    obs, mask = synthetic_batch(2)
    _, pending, cargo = split_obs(obs)
    pending.zero_()
    cargo.zero_()
    cargo[1, 0, 0] = cargo[1, 0, 1 + N_TYPES + 2] = 1
    mask.zero_()
    mask[:, -1] = True
    captured = []
    hook = net.deliver_head.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    logits, values = net(obs, mask)
    hook.remove()
    assert torch.isfinite(logits).all() and torch.isfinite(values).all()
    assert torch.equal(captured[0][0, :, 128:258], torch.zeros(6, 130))


def test_box_destination_changes_deliver_logit():
    net = SetDestNet()
    # Isolate the cargo count input so this cannot pass through shared context alone.
    with torch.no_grad():
        for parameter in net.deliver_head.parameters():
            parameter.zero_()
        net.deliver_head[0].weight[0, 256] = 1
        net.deliver_head[2].weight[0, 0] = 1
    obs, mask = synthetic_batch(1)
    mask.fill_(True)
    changed = obs.clone()
    cargo = split_obs(changed)[2]
    cargo[:, 0, 1 + N_TYPES:1 + N_TYPES + len(FACILITIES)] = 0
    cargo[:, 0, 1 + N_TYPES + 4] = 1
    with torch.no_grad():
        before = net(obs, mask)[0]
        after = net(changed, mask)[0]
    assert after[0, DELIVER0 + 1] < before[0, DELIVER0 + 1]
    assert after[0, DELIVER0 + 4] > before[0, DELIVER0 + 4]


def test_destination_checkpoint_round_trip_and_policy(tmp_path):
    net = SetDestNet().eval()
    path = tmp_path / "net.pt"
    save_checkpoint(net, "set_dest", path, {"note": "untrained"})
    loaded, info = load_checkpoint(path)
    assert isinstance(loaded, SetDestNet) and not loaded.training
    assert info == {"kind": "set_dest", "env_version": ENV_VERSION, "note": "untrained"}
    obs, mask = synthetic_batch()
    with torch.no_grad():
        expected, actual = net(obs, mask), loaded(obs, mask)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    policy = NetworkPolicy(loaded)
    assert policy(None, obs[0].numpy(), mask[0].numpy()) == actual[0][0].argmax().item()
    assert np.allclose(policy.probabilities(obs[0].numpy(), mask[0].numpy()),
                       torch.softmax(actual[0][0], 0).numpy())


def test_destination_ppo_update_on_synthetic_buffer():
    torch.manual_seed(0)
    net = SetDestNet()
    obs, mask = synthetic_batch(8)
    with torch.no_grad():
        logits, value = net(obs, mask)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
    shape = (4, 2)
    buf = RolloutBuffer(obs.numpy().reshape(*shape, OBS_DIM),
                        mask.numpy().reshape(*shape, N_ACTIONS), action.numpy().reshape(shape),
                        dist.log_prob(action).numpy().reshape(shape), value.numpy().reshape(shape),
                        np.arange(8, dtype=np.float32).reshape(shape) / 8,
                        np.zeros(shape, bool), np.zeros(2, np.float32))
    adv, ret = compute_gae(buf.reward, buf.value, buf.done, buf.last_value)
    before = net.deliver_head[0].weight.detach().clone()
    opt = torch.optim.Adam(net.parameters(), lr=3e-4, eps=1e-5)
    stats = ppo_update(net, opt, buf, adv, ret, epochs=1, minibatch=8,
                       rng=np.random.default_rng(0))
    assert all(np.isfinite(v) for v in stats.values())
    assert not torch.equal(before, net.deliver_head[0].weight)


def test_destination_keeps_existing_heads_and_registry():
    assert NETWORKS["set"] is SetNet
    old, new = SetNet(), SetDestNet()
    for name in ("pending_enc", "cargo_enc", "global_enc", "attn", "pick_head", "value_head"):
        assert {k: v.shape for k, v in getattr(old, name).state_dict().items()} == {
            k: v.shape for k, v in getattr(new, name).state_dict().items()}
