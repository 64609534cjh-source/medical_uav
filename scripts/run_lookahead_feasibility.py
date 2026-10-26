"""Fixed small Dev-tune compute feasibility; no tuning/confirmation/Test API."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from medical_uav.agents.destination_inference import DestinationInferencePolicy
from medical_uav.agents.networks import build_network
from medical_uav.env import MedicalUAVEnv
from medical_uav.env.actions import decode
from medical_uav.env.config import EnvConfig
from medical_uav.metrics import episode_record
from medical_uav.research.lookahead import PublicLookaheadPolicy
from medical_uav.training.io import clean_json
from medical_uav.training.trainer import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def quantiles(values):
    return [float(np.quantile(values, q)) for q in (.5, .95)]+[max(values)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('fresh output directory required')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('commit source/protocol before feasibility evaluation')
    protocol_path = ROOT / 'configs/gen4_lookahead_feasibility.json'
    spec = json.loads(protocol_path.read_text())
    registry = json.loads((ROOT / 'configs/gen4_scenario_registry.json').read_text())
    if (spec['split'] != 'dev_tune'
            or not set(spec['demand_seeds']) <= set(registry['new_development']['dev_tune']['demand_seeds'])):
        raise RuntimeError('only registered Dev-tune days allowed')
    for path, expected in spec['frozen_source'].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise RuntimeError('frozen source mismatch: '+path)
    manifest = json.loads((ROOT / 'artifacts/checkpoint_manifest.json').read_text())
    entry = next(e for e in manifest['models'] if e['method'] == 'destination' and e['learner_seed'] == 1)
    checkpoint = ROOT / entry['path']
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != entry['sha256']:
        raise RuntimeError('frozen checkpoint mismatch')
    torch.set_num_threads(1)
    state = load_checkpoint(checkpoint)
    model = build_network(state['architecture'])
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    config = EnvConfig(**state['env_config'])
    args.output_dir.mkdir(parents=True)
    cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                if line.startswith('model name')), platform.processor())
    output = {'experiment_id': spec['experiment_id'], 'source_commit': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'protocol_sha256': hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        'model': entry, 'python': platform.python_version(), 'torch': str(torch.__version__),
        'cpu': cpu, 'threads': 1, 'variants': {},
        'scope': 'one frozen learner, five predetermined Dev-tune days x three loads; '
                 'engineering feasibility only, not a multi-seed improvement claim'}
    started = time.perf_counter()
    # Sequential variants: no concurrent project simulation/training jobs may
    # compete with a policy whose decision rule includes a wall-clock budget.
    for variant in spec['variants']:
        tick = time.perf_counter()
        policy = (DestinationInferencePolicy(model) if variant == 'reference' else PublicLookaheadPolicy(
            model, config, scenarios=spec['scenarios'], horizon=spec['horizon_minutes'],
            value_tail=variant == 'rollout_value', budget_ms=spec['budget_ms'],
            planner_seed=spec['planner_seed'])).eval()
        setup_seconds = time.perf_counter()-tick
        records, plans, latencies = [], [], []
        tick = time.perf_counter()
        env = MedicalUAVEnv(config=config)
        try:
            for load in spec['loads']:
                for day in spec['demand_seeds']:
                    obs, info = env.reset(options={'demand_seed': day, 'load': load})
                    counts, minutes, decisions = Counter(), Counter(), 0
                    while True:
                        before = time.perf_counter()
                        with torch.inference_mode():
                            action = int(policy.act(torch.from_numpy(obs), torch.from_numpy(info['action_mask']), True)[0])
                        elapsed_ms = (time.perf_counter()-before)*1000
                        latencies.append(elapsed_ms)
                        if not info['action_mask'][action]:
                            raise RuntimeError('illegal live action')
                        if variant != 'reference':
                            plans.append({'day': day, 'load': load, 'decision': decisions, 't': env.sim.t,
                                          'action': action, 'reference_action': policy.last_original_action,
                                          'result': policy.last_outcome, 'latency_ms': elapsed_ms,
                                          'scenario_signatures': list(policy.last_scenario_signatures),
                                          'completed_scores': list(policy.last_scores)})
                        obs, reward, done, truncated, info = env.step(action)
                        if info['invalid_action'] or truncated or not np.isfinite(reward):
                            raise RuntimeError('invalid live transition')
                        env.sim.check_invariant()
                        kind = decode(action)[0]
                        counts[kind] += 1
                        minutes[kind] += info['elapsed']
                        decisions += 1
                        if decisions > 10000:
                            raise RuntimeError('complete day exceeded decision cap')
                        if done:
                            row = episode_record(env, load, decisions, counts, minutes, 1)
                            row.update(policy=variant, split='dev-tune')
                            records.append(row)
                            break
        finally:
            env.close()
        result = {'episodes': clean_json(records), 'wall_seconds': time.perf_counter()-tick,
                  'setup_seconds': setup_seconds, 'latency_ms_p50_p95_max': quantiles(latencies),
                  'planning': policy.planning_summary() if variant != 'reference' else None,
                  'decision_plans': plans, 'raw_latency_ms': latencies}
        with (args.output_dir/(variant+'.json.gz')).open('wb') as stream, gzip.GzipFile(
                filename='', fileobj=stream, mode='wb', mtime=0) as zipped:
            zipped.write(json.dumps(result, separators=(',', ':'), allow_nan=False).encode())
        output['variants'][variant] = {k: result[k] for k in ('wall_seconds', 'setup_seconds',
                                                            'latency_ms_p50_p95_max', 'planning')}
        print(variant, len(records), 'complete Dev days', result['wall_seconds'], 'seconds', flush=True)
    output['wall_seconds'] = time.perf_counter()-started
    output['status'] = 'complete'
    (args.output_dir/'manifest.json').write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
