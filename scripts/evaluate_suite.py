"""Evaluate only predeclared models/days; a separate protocol controls each stage."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from medical_uav.training.io import write_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    repo = Path(__file__).resolve().parents[1]
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('choose a fresh output directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if protocol['split'] in ('test', 'test-v2', 'test-v3') and not protocol.get('final_candidates_frozen'):
        raise ValueError('Test requires a frozen final-comparison protocol')
    for path, expected in protocol.get('frozen_files', {}).items():
        assert digest(repo / path) == expected, path
    manifest = {'protocol': protocol, 'protocol_sha256': digest(args.protocol),
                'source_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
                'status': 'running', 'completed': []}
    write_json(args.output_dir / 'manifest.json', manifest)
    started = time.perf_counter()

    def run(entry):
        checkpoint = entry.get('checkpoint')
        if checkpoint:
            assert digest(repo / checkpoint) == entry['checkpoint_sha256']
        command = [sys.executable, str(repo / 'scripts/evaluate.py'), '--split', protocol['split'],
                   '--policies', *entry['policies'], '--threads', '1', '--learner-seed', str(entry.get('learner_seed',0)),
                   '--output', str((args.output_dir / (entry['name'] + '.json')).resolve())]
        if checkpoint:
            command += ['--checkpoint', str((repo / checkpoint).resolve())]
        if protocol.get('failure_diagnostics'):
            command += ['--failure-diagnostics']
        if protocol.get('congestion'):
            command += ['--congestion']
        if protocol.get('diagnostics'):
            command += ['--diagnostics']
        with (args.output_dir / (entry['name'] + '.log')).open('w') as log:
            subprocess.run(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True)
        return entry['name']

    try:
        with ThreadPoolExecutor(max_workers=protocol.get('jobs',5)) as pool:
            futures = [pool.submit(run, entry) for entry in protocol['entries']]
            for future in as_completed(futures):
                name = future.result(); manifest['completed'].append(name)
                manifest['wall_seconds'] = time.perf_counter() - started
                write_json(args.output_dir / 'manifest.json', manifest)
                print('Completed',name,flush=True)
    except Exception as error:
        manifest.update(status='failed',error=repr(error))
        write_json(args.output_dir / 'manifest.json',manifest)
        raise
    manifest.update(status='complete',wall_seconds=time.perf_counter()-started)
    write_json(args.output_dir / 'manifest.json',manifest)


if __name__ == '__main__':
    main()
