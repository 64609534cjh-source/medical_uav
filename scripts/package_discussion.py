"""Maintainer: package existing frozen evidence; does not run training or Test."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = 'd050c4c100b86bdb6676d57f400a24c9bc80e66b'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive(path, files):
    """Deterministic archive retaining original file bytes and relative paths."""
    with path.open('wb') as target, gzip.GzipFile(fileobj=target, mode='wb', mtime=0, filename='') as zipped:
        with tarfile.open(fileobj=zipped, mode='w') as tar:
            for file in sorted(set(files)):
                data = file.read_bytes()
                entry = tarfile.TarInfo(file.relative_to(ROOT).as_posix())
                entry.size = len(data)
                entry.mode = 0o644
                tar.addfile(entry, io.BytesIO(data))


def main():
    out = ROOT / 'artifacts'
    out.mkdir(exist_ok=True)
    entries, daily, models, runs = {}, [], [], set()
    for protocol, folder in [('final_test_protocol.json', 'final_test'),
                             ('gen2_test_v2_protocol.json', 'gen2_test_v2'),
                             ('gen3_test_v3_protocol.json', 'gen3_test_v3')]:
        spec = json.loads((ROOT / 'configs' / protocol).read_text())
        for entry in spec['entries']:
            raw = ROOT / 'outputs' / folder / (entry['name'] + '.json')
            if not raw.is_file():
                raise FileNotFoundError(raw)
            daily.append(raw)
            if not entry.get('checkpoint'):
                continue
            path = ROOT / entry['checkpoint']
            digest = sha(path)
            if digest != entry['checkpoint_sha256']:
                raise ValueError('frozen model hash mismatch: ' + str(path))
            if entry['checkpoint'] in entries:
                continue
            state = torch.load(path, weights_only=True, map_location='cpu')
            method = 'm5' if entry['method'] == 's1' else entry['method']
            config_bytes = json.dumps(state['config'], sort_keys=True, separators=(',', ':')).encode()
            entries[entry['checkpoint']] = {
                'method': method, 'learner_seed': entry['learner_seed'],
                'path': entry['checkpoint'], 'sha256': digest,
                'training_source_commit': state['source_commit'],
                'training_source_manifest_sha256': state['source_manifest_sha256'],
                'config_sha256_canonical_json': hashlib.sha256(config_bytes).hexdigest(),
                'architecture': state['architecture'], 'selected_transitions': state['steps'],
                'selection': 'full Dev equal-load mean daily on-time; reward tie-break; earliest exact tie',
            }
            models.append(path)
            runs.add(path.parent)
    evidence = list(daily)
    for run in runs:
        for name in ['config.json', 'source_manifest.json', 'summary.json', 'evaluations.jsonl',
                     'evaluation_episodes.jsonl', 'updates.jsonl']:
            path = run / name
            if path.is_file():
                evidence.append(path)
    for folder in ['gen3_baseline_diagnostics', 'gen3_urgency_devextra', 'gen3_coordinated_devextra']:
        evidence.extend((ROOT / 'outputs' / folder).glob('*_s[1-5].json'))
    for name, files in [('selected_models.tar.gz', models), ('daily_evidence.tar.gz', evidence)]:
        archive(out / name, files)
    protected = ['environment.yml', 'requirements-lock.txt', 'medical_uav/env/config.py',
                 'medical_uav/env/observation.py', 'medical_uav/env/actions.py',
                 'medical_uav/env/masks.py', 'medical_uav/env/simulator.py',
                 'medical_uav/agents/destination_inference.py']
    manifest = {
        'format_version': 1, 'baseline_source_commit': BASE,
        'inference_source_commit': BASE,
        'default_method': 'destination', 'demo_learner_seed': 1,
        'default_inference': 'factored_guarded',
        'schema': {'observation': 403, 'actions': 26, 'pending_slots': 12, 'cargo_slots': 3},
        'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(models + evidence))},
        'source_files': {name: sha(ROOT / name) for name in protected},
        'models': list(entries.values()),
        'archives': {name: sha(out / name) for name in ['selected_models.tar.gz', 'daily_evidence.tar.gz']},
        'scope': 'Original selected checkpoint bytes and existing daily/log evidence. '
                 'All three historical Tests are consumed. Recomputing tables does not evaluate new policies. '
                 'Intermediate/last checkpoints and full training episode logs remain in local outputs; '
                 'this is a selected-evidence archive, not an archive of every intermediate run file.',
    }
    (out / 'checkpoint_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    checksum_files = ['checkpoint_manifest.json', 'selected_models.tar.gz', 'daily_evidence.tar.gz']
    (out / 'SHA256SUMS').write_text(''.join(f'{sha(out / n)}  {n}\n' for n in checksum_files))
    print(json.dumps({'models': len(models), 'evidence_files': len(set(evidence)),
                      'archive_bytes': {n: (out / n).stat().st_size for n in manifest['archives']}}, indent=2))


if __name__ == '__main__':
    main()
