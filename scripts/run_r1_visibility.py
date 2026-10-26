"""Replay frozen models on legacy Dev-extra with an additive visibility observer."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch

from medical_uav.agents.networks import build_network
from medical_uav.env.config import EnvConfig
from medical_uav.evaluation.evaluate import evaluate
from medical_uav.research.visibility import VisibilityAudit
from medical_uav.training.trainer import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=('destination', 'm5'), required=True)
    parser.add_argument('--seed', type=int, choices=range(1, 6), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = ROOT / f'outputs/gen4_r1_visibility/{args.method}_s{args.seed}.json.gz'
    if output.exists():
        parser.error('output exists; preserve the original replay')
    protocol = ROOT / 'configs/gen4_r1_protocol.json'
    spec = json.loads(protocol.read_text())
    if spec['split'] != 'dev-extra' or spec['demand_seeds'] != list(range(100040, 100140)):
        raise RuntimeError('R1 permits only the frozen legacy development list')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('commit diagnostic source and protocol before formal replay')
    for path, expected in spec['frozen_source'].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise RuntimeError('frozen source mismatch: ' + path)
    manifest = json.loads((ROOT / 'artifacts/checkpoint_manifest.json').read_text())
    nets, states = {}, {}
    for method in ('destination', 'm5'):
        entry = next(e for e in manifest['models'] if e['method'] == method and e['learner_seed'] == args.seed)
        path = ROOT / entry['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise RuntimeError('frozen model hash mismatch')
        state = load_checkpoint(path)
        net = build_network(state['architecture'])
        net.load_state_dict(state['model_state_dict'])
        net.eval()
        nets[method], states[method] = net, state
    other = nets['m5' if args.method == 'destination' else 'destination']
    started = time.perf_counter()
    rows, summary = evaluate(nets[args.method], ('ppo',), spec['demand_seeds'], 'dev-extra',
                             ('low', 'medium', 'high'), args.seed,
                             env_config=EnvConfig(**states[args.method]['env_config']), diagnostics=True,
                             diagnostic_factory=lambda: VisibilityAudit(other))
    old = json.loads((ROOT / f'outputs/gen3_baseline_diagnostics/{args.method}_s{args.seed}.json').read_text())['episodes']
    old = {(r['episode_load'], r['demand_seed']): r for r in old}
    for row in rows:
        clean = dict(row)
        clean['diagnostics'] = dict(row['diagnostics'])
        clean['diagnostics'].pop('visibility_audit')
        if clean != old[row['episode_load'], row['demand_seed']]:
            raise RuntimeError('diagnostic observer changed an archived day record')
    payload = {'method': args.method, 'learner_seed': args.seed,
               'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(),
               'unchanged_archived_days': len(rows), 'wall_seconds': time.perf_counter()-started,
               'episodes': rows, 'summary': summary}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as stream, gzip.GzipFile(filename='', fileobj=stream, mode='wb', mtime=0) as zipped:
        zipped.write(json.dumps(payload, separators=(',', ':'), allow_nan=False).encode())
    print(json.dumps({k: payload[k] for k in ('method', 'learner_seed', 'unchanged_archived_days', 'wall_seconds')}))


if __name__ == '__main__':
    main()
