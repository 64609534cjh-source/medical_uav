import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from medical_uav.agents.bc import masked_imitation_loss, train_imitation
from medical_uav.agents.dagger import aggregate_datasets, collect_student_data, day_schedule
from medical_uav.agents.networks import MLPNet
from medical_uav.agents.policy import NetworkPolicy, load_checkpoint, save_checkpoint
from medical_uav.baselines.horizon import RollingHorizon
from medical_uav.env import EnvConfig, MedicalUAVEnv
from medical_uav.env.actions import WAIT
from medical_uav.splits import CONFIRM2_SEEDS, DAGGER_SEEDS, TRAINING_SEEDS


def wait_policy(env, obs, mask):
    return WAIT


def test_labels_follow_exact_student_states_and_do_not_execute_teacher():
    cfg = EnvConfig(release_window=20)
    teacher = RollingHorizon()
    data = collect_student_data(wait_policy, [709905], config=cfg, teacher=teacher)
    env = MedicalUAVEnv(config=cfg)
    obs, info = env.reset(options={"demand_seed": 709905})
    disagreements = 0
    for i, (action, label) in enumerate(zip(data["action"], data["label"])):
        np.testing.assert_array_equal(data["obs"][i], obs)
        np.testing.assert_array_equal(data["mask"][i], info["action_mask"])
        assert action == WAIT
        assert label == teacher(env)
        disagreements += action != label
        obs, _, done, _, info = env.step(int(action))
    assert done and disagreements > 0
    assert data["mask"][np.arange(len(data["label"])), data["label"]].all()


def test_process_labels_match_serial_and_replay_is_self_labelled():
    cfg = EnvConfig(release_window=20)
    serial = collect_student_data(wait_policy, [709905], config=cfg)
    parallel = collect_student_data(wait_policy, [709905], config=cfg, processes=2)
    for key in serial:
        np.testing.assert_array_equal(serial[key], parallel[key])
    replay = collect_student_data(wait_policy, [709950], loads=("low", "high"),
                                  config=cfg, label_with_teacher=False)
    np.testing.assert_array_equal(replay["action"], replay["label"])
    assert set(replay["load"]) == {0, 2}


def test_aggregation_retains_prior_iterations_and_replay():
    replay = collect_student_data(wait_policy, [709950], loads=("low", "high"),
                                  config=EnvConfig(release_window=1), label_with_teacher=False)
    first = {k: v[:1] for k, v in replay.items()}
    second = {k: v[-1:] for k, v in replay.items()}
    result = aggregate_datasets(replay, first, second)
    assert len(result["action"]) == len(replay["action"]) + 2
    for key in result:
        np.testing.assert_array_equal(result[key][:len(replay[key])], replay[key])
    with pytest.raises(ValueError):
        aggregate_datasets(replay, {"obs": replay["obs"]})


def test_imitation_uses_label_not_executed_action_and_masks_gradients():
    logits = torch.tensor([[0., 1000., 0.]], requires_grad=True)
    mask = torch.tensor([[True, False, True]])
    loss = masked_imitation_loss(logits, mask, torch.tensor([2]))
    assert loss.item() == pytest.approx(np.log(2))
    loss.backward()
    assert logits.grad[0, 1] == 0 and logits.grad[0, 2] < 0
    with pytest.raises(ValueError, match="label"):
        masked_imitation_loss(logits, mask, torch.tensor([1]))
    net = MLPNet()
    data = {"obs": np.zeros((8, net.obs_dim), np.float32),
            "mask": np.ones((8, 26), bool), "action": np.zeros(8, np.int64),
            "label": np.full(8, WAIT, np.int64)}
    critic = {k: v.clone() for k, v in net.critic.state_dict().items()}
    losses = train_imitation(net, data, epochs=4, lr=.01, batch_size=8)
    assert losses[-1] < losses[0]
    assert NetworkPolicy(net)(None, data["obs"][0], data["mask"][0]) == WAIT
    assert all(torch.equal(v, critic[k]) for k, v in net.critic.state_dict().items())


def test_registered_schedules_are_disjoint_and_confirmation_is_new():
    used = set()
    for seed in TRAINING_SEEDS:
        medium, replay = day_schedule(seed, 3, 16)
        assert len(medium) == 3 and all(len(block) == 16 for block in medium)
        days = set(sum(medium, []) + replay)
        assert len(days) == 64 and days <= set(DAGGER_SEEDS)
        assert not used & days
        used |= days
    assert CONFIRM2_SEEDS == range(600000, 600300)
    for block in (range(0, 100340), range(200000, 200300), range(300000, 300100),
                  range(400000, 400100), range(500000, 500300), DAGGER_SEEDS):
        assert not set(CONFIRM2_SEEDS) & set(block)
    assert not used & set(range(709900, 710000))
    with pytest.raises(ValueError):
        day_schedule(11, 100, 100)


def test_training_cli_smoke_preserves_checkpoint_and_aggregate_format(tmp_path):
    initial = tmp_path / "initial.pt"
    save_checkpoint(MLPNet(), "mlp", initial, {"reward": "v2_expiry05"})
    out = tmp_path / "student"
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/train_dagger.py", "--init", str(initial), "--seed", "11",
         "--iterations", "1", "--days-per-iteration", "1", "--processes", "2",
         "--smoke", "--out", str(out)], cwd=root, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(root)}, timeout=90)
    assert result.returncode == 0, result.stderr
    checkpoints = sorted(out.glob("ckpt_*.pt"))
    assert len(checkpoints) == 2
    net, info = load_checkpoint(checkpoints[-1])
    assert net.obs_dim == 452 and info["iteration"] == 1
    assert info["teacher"] == "visible_horizon" and info["teacher_reward"] == "v2"
    meta = json.loads((out / "run.json").read_text())
    assert meta["smoke"] and len(meta["init_sha256"]) == 64
    with np.load(out / "dataset_01.npz") as data:
        assert set(data["load"]) == {0, 1, 2}
        assert info["decisions"] == len(data["label"])
        assert set(data["seed"]) <= set(range(709900, 710000))
