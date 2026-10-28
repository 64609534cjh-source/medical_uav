import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from medical_uav.env.config import ENV_VERSION, ORDER_TYPES
from medical_uav.splits import FRESH_SEEDS, TRAINING_SEEDS


def load_script(relative):
    path = Path(__file__).resolve().parents[1] / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analysis = load_script("docs/superpowers/experiments/2026-10-27-final/improvement_analysis.py")


def selection(arm="dest", seeds=TRAINING_SEEDS):
    return {"runs": {f"runs/{arm}_set_s{s}": {
        "selected": f"runs/{arm}_set_s{s}/ckpt_06000640.pt"} for s in seeds}}


def fake_rows(methods=("new", "frozen", "slack"), days=FRESH_SEEDS, maps=(0,), horizon=None):
    rows = []
    for method in methods:
        for load in (analysis.LOADS[1:] if method == horizon else analysis.LOADS):
            for map_seed in maps:
                for day in days:
                    better = method.startswith("dest_set") or method == "new"
                    on, served = (85, 95) if better else (80, 90)
                    row = {"method": method, "load": load, "map": map_seed, "seed": day,
                           "env_version": ENV_VERSION, "on_time_rate": on/100,
                           "served_rate": served/100, "expiry_rate": (100-served)/100}
                    for t in ORDER_TYPES:
                        row.update({f"released_{t.name}": 100, f"on_time_count_{t.name}": on,
                                    f"served_count_{t.name}": served, f"expired_count_{t.name}": 100-served})
                    rows.append(row)
    return rows


def write_rows(path, rows):
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_analysis_complete_synthetic_confirmation(tmp_path, monkeypatch):
    # Shrink fixtures/replicates, retaining all five training seeds and all reference families.
    monkeypatch.setattr(analysis, 'FRESH_SEEDS', range(500000, 500003))
    paths = {'dest': analysis.entrants(selection(), 'dest'),
             'frozen': analysis.frozen_paths(), 'mm': analysis.frozen_paths(True)}
    choice = {'selection_only': True, 'ready_for_fresh': True, 'completed': ['dest'],
              'candidate': 'dest', 'paths': paths}
    methods = [m for ps in paths.values() for m in analysis.checkpoint_labels(ps)]
    methods += ['slack', 'rollout', 'scenario16', 'horizon']
    path = tmp_path / 'synthetic.csv'
    write_rows(path, fake_rows(methods, analysis.FRESH_SEEDS, horizon='horizon'))
    result = analysis.confirm(choice, [path], n_boot=4)
    assert result['targets']['all_three']
    assert result['absolute']['dest']['medium+high']['plasma/served'] == pytest.approx(.95)
    ci = result['contrasts']['frozen']['medium+high']['all/on_time']
    assert ci['paired_days'] == pytest.approx([.05]*3)
    assert ci['two_way'] == pytest.approx([.05]*3)
    assert len(result['per_seed']['dest']['high']['all/expired']) == 5
    assert result['factor_contrasts']['dest']['reference'] == 'frozen'
    assert result['factor_contrasts']['dest']['endpoints']['high']['plasma/expired']['two_way'] == pytest.approx([-.05]*3)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('problem', ['missing', 'duplicate', 'old_day', 'unknown_method', 'version',
                                    'nan', 'counts', 'different_releases', 'no_served', 'rate_mismatch'])
def test_rejects_invalid_comparison(tmp_path, problem):
    rows = fake_rows(days=[500000])
    row = rows[0]
    if problem == 'missing': rows.pop()
    elif problem == 'duplicate': rows.append(row.copy())
    elif problem == 'old_day': row['seed'] = 12345  # invalid label, no episode
    elif problem == 'unknown_method': row['method'] = 'unknown'
    elif problem == 'version': row['env_version'] = 'v1'
    elif problem == 'nan': row['on_time_rate'] = float('nan')
    elif problem == 'counts': row['on_time_count_plasma'] = 101
    elif problem == 'different_releases':
        row['released_plasma'] = 101
        row['expired_count_plasma'] += 1
    elif problem == 'no_served': row['served_count_plasma'] = ''
    else: row['served_rate'] = .92
    path = tmp_path/'invalid.csv'
    write_rows(path, rows)
    with pytest.raises(ValueError):
        analysis.read_rows([path], ['new', 'frozen', 'slack'], [500000])


def test_missing_seed_failed_selection_and_wrong_run():
    data = selection()
    data['runs'].pop('runs/dest_set_s55')
    with pytest.raises(ValueError, match='exactly'):
        analysis.entrants(data, 'dest')
    data = selection()
    data['runs']['runs/dest_set_s11']['selected'] = None
    with pytest.raises(ValueError, match='Selection failure'):
        analysis.entrants(data, 'dest')
    data['runs']['runs/dest_set_s11']['selected'] = 'runs/other/ckpt.pt'
    with pytest.raises(ValueError, match='registered run'):
        analysis.entrants(data, 'dest')


def test_type_estimator_pools_counts_and_bootstraps_whole_days():
    num = np.array([[[[10., 0.]]]])
    den = np.array([[[[10., 100.]]]])
    assert analysis.estimate((num, den)) == pytest.approx(10/110)
    zero = (np.zeros_like(num), den)
    ci = analysis.bootstrap((num, den), zero, n_boot=30)
    assert ci['paired_days'][0] == pytest.approx(10/110)
    assert ci['paired_days'][1:] == pytest.approx([0., 1.])


def test_two_way_bootstrap_resamples_both_axes():
    differences = np.array([[.01, .03, .06], [-.02, .01, .04]])
    rng = np.random.default_rng(0)
    boot = [differences[np.ix_(rng.integers(2, size=2), rng.integers(3, size=3))].mean()
            for _ in range(100)]
    actual = analysis.two_way_ci(differences, n_boot=100)
    assert actual == pytest.approx([differences.mean(), *np.percentile(boot, [2.5, 97.5])])
    a = np.repeat(np.array([-.02, .02])[:, None, None, None], 30, axis=3)
    ci = analysis.bootstrap((a, np.ones_like(a)), (np.zeros_like(a), np.ones_like(a)), n_boot=100)
    assert ci['paired_days'][1:] == [0., 0.]
    assert ci['two_way'][1] < 0 < ci['two_way'][2]


def test_reward_selection_boundaries_and_ties():
    v2 = {'on_time': .8, 'expired': .1}
    assert analysis.choose_reward({'v2': v2}) == 'v2'
    assert analysis.choose_reward({'v2': v2, 'v2_expiry03': {'on_time': .795, 'expired': .08},
                                   'v2_expiry05': {'on_time': .7949, 'expired': .01}}) == 'v2_expiry03'
    assert analysis.choose_reward({'v2': v2, 'v2_expiry05': {'on_time': .9, 'expired': .1}}) == 'v2'
    assert analysis.choose_reward({'v2': v2, 'v2_expiry03': {'on_time': .8, 'expired': .07},
                                   'v2_expiry05': {'on_time': .8, 'expired': .07}}) == 'v2_expiry03'


@pytest.mark.parametrize('difference,broken,drop', [(-.02, False, False), (-.02001, False, True),
                                                 (.03, True, True), (0, False, False)])
def test_pilot_strict_drop_boundary(difference, broken, drop):
    assert analysis.pilot_drop(difference, broken) is drop


def test_target_pass_fail_boundaries():
    def ci(point, lower): return {k: [point, lower, .1] for k in ('paired_days', 'two_way')}
    on = {'horizon': ci(.01, .001), 'frozen': ci(.02, .002)}
    assert analysis.target_readings(on, on, ci(.01, .001))['all_three']
    on['horizon'] = ci(0, -.005)
    readings = analysis.target_readings(on, on, ci(.0099, .001))
    assert not readings['reach_horizon_with_0.5pp_tolerance']
    assert not readings['expiry_reduction_at_least_1pp'] and not readings['beat_all_on_time']
    on['horizon'] = ci(0, -.00499)
    assert analysis.target_readings(on, on, ci(.01, 0))['reach_horizon_with_0.5pp_tolerance']
    assert not analysis.target_readings(on, on, ci(.01, 0))['expiry_reduction_at_least_1pp']


def test_selection_export_routes_C_only_to_dev_maps(monkeypatch):
    calls = []
    monkeypatch.setattr(analysis.subprocess, 'run', lambda cmd, check: calls.append(cmd))
    analysis.export_selection('notable', selection('notable', (11,)), 'runs')
    assert calls[0][1] == 'scripts/map_stress.py'
    assert calls[0][calls[0].index('--maps')+1] == '1-10'
    assert calls[0][calls[0].index('--days')+1] == '100'
    assert any('mm_set_s11' in x for x in calls[0])
    analysis.export_selection('dest', selection('dest', (11,)), 'runs')
    assert calls[1][calls[1].index('--seeds')+1] == 'select'


def test_choice_rejects_fresh_access_before_combined_selection():
    with pytest.raises(ValueError, match='Freeze'):
        analysis.validate_choice({'selection_only': True, 'ready_for_fresh': False})


def test_fresh_cli_routes_exact_days_without_policy_execution(tmp_path, monkeypatch):
    import sys

    script = load_script("scripts/evaluate.py")
    jobs = []

    def fake_job(job):
        jobs.append(job)
        method, _, _, load, days = job
        return [{"method": method, "load": load, "seed": days[0]}]

    monkeypatch.setattr(script, "run_job", fake_job)
    monkeypatch.setattr(script, "print_summary", lambda rows, loads: None)
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--methods", "slack", "--seeds", "fresh",
                                     "--out", str(tmp_path / "mock.csv")])
    script.main()
    assert len(jobs) == 900
    assert {job[-1][0] for job in jobs} == set(FRESH_SEEDS)
    assert {job[-2] for job in jobs} == set(analysis.LOADS)


def test_fresh_split_is_separate_from_every_used_day_block():
    assert FRESH_SEEDS == range(500_000, 500_300)
    for lo, hi in ((0, 100_340), (200_000, 200_300), (300_000, 300_100), (400_000, 400_100)):
        assert not set(FRESH_SEEDS).intersection(range(lo, hi))


def test_training_cli_registers_destination_network_without_running_it(monkeypatch, capsys):
    import sys

    script = load_script("scripts/train_ppo.py")
    monkeypatch.setattr(sys, "argv", ["train_ppo.py", "--help"])
    with pytest.raises(SystemExit) as exc:
        script.main()
    assert exc.value.code == 0
    assert "set_dest" in capsys.readouterr().out


@pytest.mark.parametrize('combined_score,combined_expiry,expected', [
    (.825, .07, 'combined'), (.8249, .07, 'expiry03'), (.83, .081, 'expiry03')])
def test_choice_and_single_combination_rule(monkeypatch, combined_score, combined_expiry, expected):
    values = {'frozen': {'on_time': .8, 'expired': .1},
              'expiry03': {'on_time': .82, 'expired': .08},
              'expiry05': {'on_time': .79, 'expired': .02},
              'dest': {'on_time': .805, 'expired': .1},
              'combined': {'on_time': combined_score, 'expired': combined_expiry}}
    monkeypatch.setattr(analysis, 'selection_data',
                        lambda directory, arm, seeds: (arm, [arm], ['frozen'], (0,)))
    monkeypatch.setattr(analysis, 'score', lambda rows, labels, metric: values[labels[0]][metric])
    monkeypatch.setattr(analysis, 'load_json', lambda path: selection(Path(path).stem.removeprefix('selection_')))
    manifest = {'selection_only': True, 'survivors': ['expiry03', 'expiry05', 'dest']}
    choice = analysis.choose('unused_fixture', manifest)
    assert choice['reward'] == 'v2_expiry03' and choice['combine']
    assert not choice['ready_for_fresh']
    final = analysis.choose('unused_fixture', manifest, include_combined=True)
    assert final['candidate'] == expected and final['ready_for_fresh']
    assert final['completed'].count('combined') == 1


def test_pilot_log_diagnostics_allow_empty_episode_nan_and_flat_curve(tmp_path):
    run = tmp_path/'dest_set_s11'
    run.mkdir()
    (run/'updates.csv').write_text('decisions,policy_loss,value_loss,entropy,approx_kl,clip_frac,train_on_time\n'
                                  '6000640,0.1,1.0,0.5,-0.001,0.1,nan\n')
    (run/'evals.csv').write_text('decisions,on_time_medium,on_time_high\n6000640,0.8,0.7\n')
    diag = analysis.audit_pilot_logs(tmp_path, 'dest')
    assert not diag['curve_rising']
    (run/'updates.csv').write_text((run/'updates.csv').read_text().replace('-0.001', 'nan'))
    with pytest.raises(ValueError, match='non-finite approx_kl'):
        analysis.audit_pilot_logs(tmp_path, 'dest')
