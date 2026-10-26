"""Audit locally recorded scenario IDs before reserving new development manifests."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def inspect(value, seen):
    if isinstance(value, dict):
        for key, item in value.items():
            role = {'demand_seed': 'demand', 'map_seed': 'map', 'learner_seed': 'learner'}.get(key)
            if role and isinstance(item, int) and not isinstance(item, bool):
                seen[role].add(item)
            role = {'demand_seeds': 'demand', 'train_seeds': 'demand',
                    'learner_seeds': 'learner', 'map_seeds': 'map'}.get(key)
            if role and isinstance(item, list):
                seen[role].update(x for x in item if isinstance(x, int) and not isinstance(x, bool))
            inspect(item, seen)
    elif isinstance(value, list):
        for item in value:
            inspect(item, seen)


def main():
    evidence = {}
    seen = {k: set() for k in ('demand', 'learner', 'map')}
    for folder in ['configs', 'results', 'outputs']:
        for path in sorted((ROOT / folder).rglob('*')):
            if path.suffix not in ('.json', '.jsonl') or not path.is_file():
                continue
            if any(x in path.parts for x in ('gen4_registry', 'gen4_r1_visibility', 'gen4_dev_tune')):
                continue
            if path.name in ('gen4_scenario_registry.json', 'gen4_seed_usage_audit.json', 'gen4_r1_protocol.json'):
                continue
            content = path.read_bytes()
            evidence[path.relative_to(ROOT).as_posix()] = hashlib.sha256(content).hexdigest()
            try:
                if path.suffix == '.jsonl':
                    for line in content.splitlines():
                        if line.strip():
                            inspect(json.loads(line), seen)
                else:
                    inspect(json.loads(content), seen)
            except (ValueError, UnicodeError) as error:
                raise RuntimeError('unreadable scenario evidence: ' + str(path)) from error
    proposed = {'dev_tune': list(range(100140, 100240)), 'dev_confirm': list(range(100240, 100340))}
    learners = {'explore': [11, 12], 'long_screen': [13, 14, 15], 'confirm': [21, 22, 23, 24, 25]}
    all_new = set(proposed['dev_tune'] + proposed['dev_confirm'])
    if all_new & seen['demand'] or any(s < 100000 for s in all_new):
        raise RuntimeError('proposed development IDs already recorded or overlap Train')
    if set(sum(learners.values(), [])) & seen['learner']:
        raise RuntimeError('proposed learner IDs already recorded')
    registry = {
        'format_version': 1, 'baseline_commit': 'd050c4c100b86bdb6676d57f400a24c9bc80e66b',
        'publication_base': 'a9117e3', 'map_seed': 0, 'loads': ['low', 'medium', 'high'],
        'train_seed_range_end_exclusive': [0, 100000],
        'legacy_dev': {'main': [100000, 100039], 'extra': [100040, 100139], 'status': 'used development'},
        'consumed_test_ranges_inclusive': [[200000, 200099], [300000, 300099], [400000, 400099]],
        'new_development': {name: {'demand_seeds': days,
                                  'manifest_sha256': hashlib.sha256(json.dumps(days, separators=(',', ':')).encode()).hexdigest(),
                                  'status_at_registration': 'not evaluated',
                                  'purpose': 'configuration/checkpoint selection' if name == 'dev_tune' else 'limited candidate confirmation; not Test'}
                            for name, days in proposed.items()},
        'new_learner_seeds': learners, 'new_final_test': None,
        'final_test_status': 'not generated, not opened; freeze primary candidate and protocol first',
        'audit_scope': 'All readable JSON/JSONL in local configs/results/outputs, including debugging records; '
                       'explicit demand_seed/map_seed/learner_seed fields and plural forms. '
                       'Formal legacy seed pools remain excluded even where individual IDs were not sampled. '
                       'No assertion about unrecorded external runs or arbitrary generic seed fields.',
        'selection': 'one checkpoint per learner across all loads; equal-load mean daily on-time, '
                     'then mean daily reward; earliest exact tie; no type guardrail',
        'confirmation_access': 'only after candidate/config freeze; every access recorded; no iterative tuning on confirm',
    }
    audit = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
             'evidence_files': evidence, 'observed_ids': {k: sorted(v) for k, v in seen.items()},
             'evidence_file_count': len(evidence), 'new_manifest_ids_absent_from_scanned_records': True,
             'scope': registry['audit_scope']}
    for name, data in [('gen4_scenario_registry.json', registry)]:
        (ROOT / 'configs' / name).write_text(json.dumps(data, indent=2) + '\n')
    (ROOT / 'results/gen4_seed_usage_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({'files_audited': len(evidence), 'unique_observed_ids': {k: len(v) for k, v in seen.items()},
                      'registered_dev_days_each': 100, 'new_final_test': None}))


if __name__ == '__main__':
    main()
