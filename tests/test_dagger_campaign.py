import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from medical_uav.env.config import ENV_VERSION, ORDER_TYPES
from medical_uav.splits import CONFIRM2_SEEDS, TRAINING_SEEDS


def load_script(relative):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(Path(relative).stem, root / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


campaign = load_script("scripts/dagger_campaign.py")


def rates(medium=.91, high=.81, expiry=.08):
    return {load: {"on_time_rate": on, "expiry_rate": expiry, "served_rate": 1 - expiry}
            for load, on in (("low", .99), ("medium", medium), ("high", high))}


def choice():
    paths = {arm: [f"runs/{arm}_set_s{s}/ckpt_00000001.pt" for s in TRAINING_SEEDS]
             for arm in ("dagger", "expiry05")}
    paths["frozen"] = campaign.analysis.frozen_paths()
    return {"selection_only": True, "ready_for_fresh": True, "candidate": "dagger",
            "completed": ["dagger", "expiry05"], "paths": paths,
            "confirmation_days": [CONFIRM2_SEEDS.start, CONFIRM2_SEEDS.stop]}


def test_selection_gates_boundaries_and_medium_priority():
    baseline = rates()
    results = {"ckpt_0": baseline, "ckpt_1": rates(.92, .805, .083),
               "ckpt_2": rates(.93, .8049), "ckpt_3": rates(.94, .81, .08301)}
    selected, audit = campaign.choose_checkpoint(results, baseline)
    assert selected == "ckpt_1"
    assert [row["eligible"] for row in audit] == [True, True, False, False]
    assert campaign.choose_checkpoint({"ckpt_0": baseline}, baseline)[0] == "ckpt_0"
    results["ckpt_1"]["medium"]["on_time_rate"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        campaign.choose_checkpoint(results, baseline)


def test_selection_ties_prefer_expiry_then_earlier():
    results = {"ckpt_2": rates(.92, .81, .08), "ckpt_1": rates(.92, .81, .08),
               "ckpt_0": rates(.92, .81, .081)}
    assert campaign.choose_checkpoint(results, rates())[0] == "ckpt_1"


def test_confirmation_reuses_analysis_on_only_confirm2(tmp_path, monkeypatch):
    import csv

    frozen = choice()
    methods = [m for paths in frozen["paths"].values()
               for m in campaign.analysis.checkpoint_labels(paths)]
    methods += ["slack", "rollout", "scenario16", "horizon"]
    days = CONFIRM2_SEEDS[:2]
    rows = []
    for method in methods:
        for load in (("medium", "high") if method == "horizon" else campaign.LOADS):
            for day in days:
                on, served = (.86, .96) if method.startswith("dagger_") else (.8, .9)
                row = {"method": method, "load": load, "seed": day, "map": 0,
                       "env_version": ENV_VERSION, "on_time_rate": on,
                       "served_rate": served, "expiry_rate": 1 - served}
                for typ in ORDER_TYPES:
                    row.update({f"released_{typ.name}": 100,
                                f"on_time_count_{typ.name}": round(on * 100),
                                f"served_count_{typ.name}": round(served * 100),
                                f"expired_count_{typ.name}": round((1 - served) * 100)})
                rows.append(row)
    path = tmp_path / "synthetic.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(campaign, "CONFIRM2_SEEDS", days)
    result = campaign.confirm(frozen, [path], n_boot=4)
    assert result["targets"]["all_three"]
    assert result["confirmation_days"] == list(days)
    assert result["contrasts"]["expiry05"]["medium+high"]["all/on_time"][
        "two_way"] == pytest.approx([.06, .06, .06])
    assert result["retention"]["expiry_no_worse_point"]
    assert result["retention"]["high_on_time_within_selection_tolerance"]
    assert set(result["literal_horizon_on_time_point"]) == {"medium", "high", "medium+high"}
    json.dumps(result, allow_nan=False)


def test_choice_rejects_missing_seeds_or_unfrozen_confirmation():
    data = choice()
    data["paths"]["dagger"].pop()
    with pytest.raises(ValueError):
        campaign.validate_choice(data)
    data = choice()
    data["ready_for_fresh"] = False
    with pytest.raises(ValueError):
        campaign.validate_choice(data)
    data = choice()
    data["confirmation_days"] = [500000, 500300]
    with pytest.raises(ValueError):
        campaign.validate_choice(data)


def test_export_uses_new_days_and_limits_horizon_to_medium_high(tmp_path, monkeypatch):
    calls = []

    def fake_job(job):
        calls.append(job)
        return [{"method": job[2], "load": job[3], "seed": job[4][0]}]

    monkeypatch.setattr(campaign, "run_job", fake_job)
    campaign.export_rows(choice(), "horizon", 1, tmp_path / "mock.csv")
    assert len(calls) == 600
    assert {job[3] for job in calls} == {"medium", "high"}
    assert {job[4][0] for job in calls} == set(CONFIRM2_SEEDS)
    assert all(job[0] == "horizon" and job[1] is None for job in calls)


def test_frozen_hash_audit_detects_changed_checkpoint(tmp_path):
    path = tmp_path / "ckpt.pt"
    path.write_bytes(b"original")
    data = {"checkpoint_sha256": {str(path): campaign.file_hash(path)}}
    campaign.audit_hashes(data)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        campaign.audit_hashes(data)
