"""Run the predeclared S1 stages, always scratch and always the complete budget."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from medical_uav.training.io import write_json
from run_m3 import check_frozen


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=Path('configs/m5_protocol.json'))
    parser.add_argument('--stage',choices=('development','extension'),required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();repo=Path(__file__).resolve().parents[1]
    protocol=json.loads(args.protocol.read_text());check_frozen(repo,protocol)
    seeds=protocol['stages'][args.stage]
    if args.output_dir.exists() and any(args.output_dir.iterdir()):parser.error('fresh output directory required')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    manifest={'protocol':protocol,'stage':args.stage,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=repo).strip(),
              'protocol_sha256':hashlib.sha256(args.protocol.read_bytes()).hexdigest(),'status':'running','completed_seeds':[]}
    write_json(args.output_dir/'manifest.json',manifest);started=time.perf_counter()
    def run(seed):
        check_frozen(repo,protocol)
        name=protocol['configurations'][str(seed)]
        assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==protocol['configuration_sha256'][name]
        config=json.loads((repo/name).read_text())
        assert config['learning_rate_schedule']=='late_linear_10pct' and config['total_steps']==1024000 and config['learner_seed']==seed
        with (args.output_dir/f'seed_{seed}.log').open('w') as log:
            subprocess.run(['/usr/bin/time','-p',sys.executable,str(repo/'scripts/train.py'),'--config',str(repo/name),
                            '--output-dir',str((args.output_dir/f'seed_{seed}').resolve())],cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((args.output_dir/f'seed_{seed}/summary.json').read_text())
        assert result['transitions']==1024000 and result['updates']==1000 and result['status']=='complete'
        return seed
    try:
        with ThreadPoolExecutor(max_workers=len(seeds)) as pool:
            for future in as_completed([pool.submit(run,seed) for seed in seeds]):
                seed=future.result();manifest['completed_seeds'].append(seed);manifest['completed_seeds'].sort()
                manifest['wall_seconds']=time.perf_counter()-started;write_json(args.output_dir/'manifest.json',manifest)
                print('Completed S1 learner seed',seed,flush=True)
    except Exception as error:
        manifest.update(status='failed',error=repr(error));write_json(args.output_dir/'manifest.json',manifest);raise
    manifest.update(status='complete',wall_seconds=time.perf_counter()-started)
    write_json(args.output_dir/'manifest.json',manifest)


if __name__=='__main__':main()
