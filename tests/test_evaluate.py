import numpy as np
import pytest

from medical_uav.baselines.heuristics import fcfs_policy, slack_aware_policy
from medical_uav.env.actions import PICK0
from medical_uav.evaluation import evaluate, heuristic, paired_ci


def test_evaluate_rows_and_columns():
    rows = evaluate({"slack": heuristic(slack_aware_policy), "fcfs": heuristic(fcfs_policy)},
                    ["low"], range(100_000, 100_002))
    assert len(rows) == 4
    keys = {"method", "load", "seed", "env_version", "on_time", "n_orders", "on_time_rate",
            "served_rate", "expiry_rate", "return", "charge_share", "on_time_plasma"}
    assert keys <= set(rows[0])
    assert {r["method"] for r in rows} == {"slack", "fcfs"}


def test_evaluate_rejects_invalid_action():
    def bad(env, obs, mask):
        return int(np.flatnonzero(~mask)[0]) if (~mask).any() else PICK0
    with pytest.raises(AssertionError):
        evaluate({"bad": bad}, ["low"], [100_000])


def test_paired_ci_constant():
    assert paired_ci(np.full(20, 0.1)) == pytest.approx((0.1, 0.1, 0.1))


def test_checkpoint_labels_are_unique():
    from medical_uav.evaluation import checkpoint_labels

    labels = checkpoint_labels(["runs/a/ckpt_01.pt", "runs/b/ckpt_01.pt", "checkpoints/bc_set.pt"])
    assert labels == ["a/ckpt_01", "b/ckpt_01", "bc_set"]
