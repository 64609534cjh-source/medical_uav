import numpy as np

from medical_uav.agents.bc import action_families, collect_demonstrations, split_by_seed, train_bc
from medical_uav.agents.networks import MLPNet
from medical_uav.baselines.heuristics import slack_aware_policy
from medical_uav.env import MedicalUAVEnv
from medical_uav.env.actions import CHARGE0, DELIVER0, PICK0, RETURN, WAIT


def test_demonstrations_match_teacher():
    data = collect_demonstrations(slack_aware_policy, ["low"], [0, 1])
    n = len(data["action"])
    assert all(len(v) == n for v in data.values()) and n > 0
    assert data["obs"].dtype == np.float32 and data["mask"].dtype == bool
    assert data["mask"][np.arange(n), data["action"]].all()
    # replaying the recorded actions of seed 0 reproduces the recorded observations
    first = data["seed"] == 0
    env = MedicalUAVEnv(load="low")
    obs, _ = env.reset(options={"demand_seed": 0})
    for recorded, a in zip(data["obs"][first], data["action"][first]):
        assert np.array_equal(obs, recorded)
        obs, *_ = env.step(int(a))


def test_split_by_seed():
    data = collect_demonstrations(slack_aware_policy, ["low", "high"], range(10))
    train, val = split_by_seed(data, val_fraction=0.2, seed=0)
    assert not set(train["seed"]) & set(val["seed"])
    assert len(set(val["seed"])) == 2 and len(set(train["seed"])) == 8
    assert len(train["action"]) + len(val["action"]) == len(data["action"])


def test_action_families():
    names = action_families(np.array([PICK0 + 3, DELIVER0 + 1, CHARGE0 + 2, RETURN, WAIT]))
    assert list(names) == ["pick", "deliver", "charge", "return", "wait"]


def test_train_bc_reduces_loss():
    data = collect_demonstrations(slack_aware_policy, ["low", "medium"], [0, 1, 2])
    train, val = split_by_seed(data, val_fraction=0.34, seed=0)
    history = train_bc(MLPNet(), train, val, epochs=3)
    assert len(history) == 3
    assert {"train_loss", "val_loss", "val_acc", "val_acc_by_family"} <= set(history[0])
    assert history[2]["train_loss"] < history[0]["train_loss"]


def test_trained_checkpoint_with_history_loads(tmp_path):
    from medical_uav.agents.policy import load_checkpoint, save_checkpoint
    data = collect_demonstrations(slack_aware_policy, ["low"], [0, 1])
    train, val = split_by_seed(data, val_fraction=0.5, seed=0)
    history = train_bc(MLPNet(), train, val, epochs=1)
    info = {"val_acc_by_family": history[0]["val_acc_by_family"]}
    save_checkpoint(MLPNet(), "mlp", tmp_path / "bc.pt", info)
    _, info = load_checkpoint(tmp_path / "bc.pt")
    assert all(type(k) is str and type(v) is float for k, v in info["val_acc_by_family"].items())


def test_new_network_is_reproducible():
    import torch

    from medical_uav.agents.bc import new_network
    a, b = new_network("set", seed=0), new_network("set", seed=0)
    assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))
