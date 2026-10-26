"""Replay four fixed Dev examples; simulator event fields are diagnostics only."""
from dataclasses import asdict
import gzip
import json
from pathlib import Path

import torch

from medical_uav.agents.networks import build_network
from medical_uav.env import MedicalUAVEnv
from medical_uav.env.actions import action_name
from medical_uav.env.config import EnvConfig
from medical_uav.env.observation import rank_pending
from medical_uav.training.trainer import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    manifest = json.loads((ROOT / 'artifacts/checkpoint_manifest.json').read_text())
    cases = json.loads((ROOT / 'results/discussion_failure_audit.json').read_text())['representatives']
    traces = []
    for method in ['m5', 'destination']:
        entry = next(m for m in manifest['models'] if m['method'] == method and m['learner_seed'] == 1)
        state = load_checkpoint(ROOT / entry['path'])
        model = build_network(state['architecture'])
        model.load_state_dict(state['model_state_dict'])
        model.eval()
        for case in cases:
            env = MedicalUAVEnv(config=EnvConfig(**state['env_config']))
            try:
                obs, info = env.reset(options={'demand_seed': case['day'], 'load': case['load']})
                steps = []
                while True:
                    sim = env.sim
                    # Only obs/mask reach the network; order IDs and physical state stay in this logger.
                    with torch.inference_mode():
                        logits, value = model(torch.from_numpy(obs))
                        action = int(model.act(torch.from_numpy(obs), torch.from_numpy(info['action_mask']), True)[0])
                    if not info['action_mask'][action]:
                        raise RuntimeError('illegal trace action')
                    record = {'t': sim.t, 'location': sim.location, 'battery': sim.battery,
                              'payload_kg': sim.load_kg(), 'cargo_ids': [o.id for o in sim.onboard],
                              'visible_ids': [o.id for o in rank_pending(sim)],
                              'observation': obs.tolist(), 'mask': info['action_mask'].tolist(),
                              'logits': logits.tolist(), 'value': float(value),
                              'action': action, 'action_name': action_name(action)}
                    before = {o.id: o.status for o in sim.orders}
                    obs, reward, terminated, truncated, info = env.step(action)
                    env.sim.check_invariant()
                    if info['invalid_action']:
                        raise RuntimeError('invalid trace transition')
                    record.update({'end_t': env.sim.t, 'reward': reward,
                                   'order_status_events': [{'id': o.id, 'before': before[o.id], 'after': o.status}
                                                           for o in env.sim.orders if before[o.id] != o.status]})
                    steps.append(record)
                    if terminated or truncated:
                        if truncated:
                            raise RuntimeError('incomplete trace')
                        break
                orders = [dict(asdict(o), deadline=o.deadline, expiry=o.expiry) for o in env.sim.orders if o.status != 'future']
                on_time = sum(o['status'] == 'delivered' and o['end_time'] <= o['deadline'] + 1e-9 for o in orders)
                expected = case['m5_rate' if method == 'm5' else 'destination_rate']
                if abs(on_time/len(orders) - expected) > 1e-12:
                    raise RuntimeError('trace differs from archived Dev result')
                traces.append({'case': case, 'method': method, 'checkpoint_sha256': entry['sha256'],
                               'steps': steps, 'orders': orders, 'complete_day': True,
                               'on_time_rate': expected, 'archived_rate_exact': True})
            finally:
                env.close()
    payload = {'scope': 'Eight paired Dev-extra trajectories; descriptive case selection; '
                        'diagnostic simulator fields never enter model observations', 'traces': traces}
    output = ROOT / 'results/discussion_traces.json.gz'
    with output.open('wb') as stream, gzip.GzipFile(filename='', fileobj=stream, mode='wb', mtime=0) as zipped:
        zipped.write(json.dumps(payload, separators=(',', ':'), allow_nan=False).encode())
    print('Saved eight complete Dev trajectories:', output)


if __name__ == '__main__':
    main()
