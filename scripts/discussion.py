"""Discussion release: verified artifacts, default Dev replay and table reproduction.

Run from an installed clone. This entry point never evaluates a Test policy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unpack():
    manifest = json.loads((ROOT / 'artifacts/checkpoint_manifest.json').read_text())
    for name, expected in manifest['archives'].items():
        archive = ROOT / 'artifacts' / name
        if digest(archive) != expected:
            raise ValueError('archive hash mismatch: ' + name)
        with tarfile.open(archive, 'r:gz') as tar:
            for entry in tar:
                path = Path(entry.name)
                if (path.is_absolute() or '..' in path.parts or not entry.isfile()
                        or entry.name not in manifest['files']):
                    raise ValueError('unexpected archive entry')
                data = tar.extractfile(entry).read()
                if hashlib.sha256(data).hexdigest() != manifest['files'][entry.name]:
                    raise ValueError('member hash mismatch: ' + entry.name)
                target = ROOT / path
                if not target.resolve().is_relative_to(ROOT):
                    raise ValueError('artifact target outside checkout')
                if target.exists():
                    if digest(target) != manifest['files'][entry.name]:
                        raise ValueError('refusing to overwrite different local evidence: ' + entry.name)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
    for name, expected in manifest['files'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('missing or changed artifact: ' + name)
    for name, expected in manifest['source_files'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('frozen source mismatch: ' + name)
    return manifest


def model_for_seed(manifest, seed):
    import torch
    from medical_uav.agents.networks import build_network
    from medical_uav.training.trainer import load_checkpoint
    entry = next(m for m in manifest['models'] if m['method'] == 'destination' and m['learner_seed'] == seed)
    state = load_checkpoint(ROOT / entry['path'])
    torch.manual_seed(seed)
    model = build_network(state['architecture'])
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    return model, state, entry


def replay(manifest, seed, day, load, output):
    import torch
    from medical_uav.agents.destination_inference import DestinationInferencePolicy
    from medical_uav.env.config import EnvConfig
    from medical_uav.evaluation.evaluate import evaluate
    from medical_uav.training.io import write_json
    torch.set_num_threads(1)
    model, state, entry = model_for_seed(manifest, seed)
    policy = DestinationInferencePolicy(model)
    rows, summary = evaluate(policy, ('ppo',), (day,), 'dev', (load,), seed,
                             env_config=EnvConfig(**state['env_config']))
    canonical, _ = evaluate(model, ('ppo',), (day,), 'dev', (load,), seed,
                            env_config=EnvConfig(**state['env_config']))
    if rows != canonical:
        raise RuntimeError('guarded/reference day mismatch')
    original = [json.loads(line) for line in (ROOT / entry['path']).parent.joinpath('evaluation_episodes.jsonl').read_text().splitlines()]
    expected = [r for r in original if r.get('transitions') == state['steps'] and r['policy'] == 'ppo'
                and r['demand_seed'] == day and r['episode_load'] == load]
    if len(expected) != 1:
        raise RuntimeError('missing original selected-checkpoint Dev day')
    expected = {k: v for k, v in expected[0].items() if k not in ('transitions', 'update_index')}
    if expected != canonical[0]:
        raise RuntimeError('replay differs from archived training-time Dev record')
    write_json(output, {'model': entry, 'inference': 'factored_guarded',
                        'fallback_states': policy.fallback_states, 'canonical_day_exact': True,
                        'archived_dev_day_exact': True, 'episodes': rows, 'summary': summary})
    print(json.dumps({'complete_days': len(rows), 'invalid_actions': rows[0]['invalid_actions'],
                      'canonical_day_exact': True, 'archived_dev_day_exact': True,
                      'fallback_states': policy.fallback_states, 'output': str(output)}, indent=2))


def tables(output):
    loads = ['low', 'medium', 'high']
    specifications = [('test', 'final_test', 'final_test_protocol.json'),
                      ('test-v2', 'gen2_test_v2', 'gen2_test_v2_protocol.json'),
                      ('test-v3', 'gen3_test_v3', 'gen3_test_v3_protocol.json')]
    lines = ['# Recomputed historical tables', '',
             'Each cohort stays separate. Mean daily rates, then equal loads. '
             '± is sample SD across five learner IDs; deterministic rules have one result. '
             'This reads archived records only; no Test policy is executed.', '']
    result = {}
    for split, folder, protocol in specifications:
        entries = json.loads((ROOT / 'configs' / protocol).read_text())['entries']
        grouped = {}
        for e in entries:
            payload = json.loads((ROOT / 'outputs' / folder / (e['name'] + '.json')).read_text())
            rows = payload['episodes']
            for policy in e['policies']:
                selected = [r for r in rows if r['policy'] == policy]
                by_day = {(r['episode_load'], r['demand_seed']): r for r in selected}
                if (len(selected) != 300 or len(by_day) != 300
                        or any(r['split'] != split or not r['complete_day'] or r['invalid_actions'] for r in selected)):
                    raise ValueError('incomplete/duplicate/invalid historical table input')
                per_load = [statistics.mean(r['on_time_rate'] for r in selected if r['episode_load'] == l) for l in loads]
                method = e['method'] if policy == 'ppo' else policy
                grouped.setdefault(method, []).append(per_load + [statistics.mean(per_load)])
        result[split] = {m: {'per_seed': values,
                             'mean': [statistics.mean(x) for x in zip(*values)],
                             'sample_sd': [statistics.stdev(x) if len(x) > 1 else None for x in zip(*values)]}
                         for m, values in grouped.items()}
        lines += [f'## {split}', '', '| Method | Low (%) | Medium (%) | High (%) | Equal (%) |', '|---|---:|---:|---:|---:|']
        for method, data in result[split].items():
            cells = [f'{100*m:.2f}' + (f' ± {100*s:.2f}' if s is not None else '')
                     for m, s in zip(data['mean'], data['sample_sd'])]
            lines.append('| ' + ' | '.join([method] + cells) + ' |')
        lines.append('')
    reference = json.loads((ROOT / 'results/gen3_test_v3.json').read_text())['methods']
    for method in ['destination', 'urgency', 'coordinated']:
        if max(abs(a-b) for a, b in zip(result['test-v3'][method]['mean'], reference[method]['mean'])) > 1e-12:
            raise RuntimeError('historical report/table mismatch')
    output.write_text('\n'.join(lines) + '\n')
    output.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Recomputed three separate historical tables:', output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('verify', 'dev', 'table'))
    parser.add_argument('--seed', type=int, choices=range(1, 6), default=1)
    parser.add_argument('--day', type=int, choices=range(100000, 100040), default=100000)
    parser.add_argument('--load', choices=('low', 'medium', 'high'), default='high')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    manifest = unpack()
    if args.command == 'verify':
        print(f"Verified {len(manifest['models'])} selected checkpoints and {len(manifest['files'])} artifact files")
        return
    output = args.output or ROOT / 'outputs/discussion' / ('dev_day.json' if args.command == 'dev' else 'tables.md')
    if output.exists() or (args.command == 'table' and output.with_suffix('.json').exists()):
        parser.error('choose fresh output files')
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == 'dev':
        replay(manifest, args.seed, args.day, args.load, output)
    else:
        tables(output)


if __name__ == '__main__':
    main()
