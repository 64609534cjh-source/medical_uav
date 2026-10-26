"""Run two preregistered five-seed scratch comparisons under frozen M5 PPO."""
import argparse,ast,hashlib,json,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from medical_uav.training.io import write_json


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_frozen(protocol):
    freeze=json.loads(Path('configs/gen3_holdout_freeze.json').read_text())
    for section in ['protected_prior_files','protected_source']:
        for p,h in freeze[section].items():assert digest(p)==h,p
    tree=ast.parse(Path('medical_uav/agents/networks.py').read_text())
    now={n.name:hashlib.sha256(ast.dump(n).encode()).hexdigest() for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
    for n,h in freeze['legacy_network_ast'].items():assert now[n]==h,n
    for p,h in protocol['frozen_files'].items():assert digest(p)==h,p
    for e in freeze['reference_models']:assert digest(e['checkpoint'])==e['checkpoint_sha256']


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,default=Path('configs/gen3_protocol.json'));parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    protocol=json.loads(args.protocol.read_text());check_frozen(protocol)
    assert not args.output_dir.exists(),'fresh output required';args.output_dir.mkdir(parents=True)
    manifest={'protocol':protocol,'protocol_sha256':digest(args.protocol),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'status':'running','completed':[]};write_json(args.output_dir/'manifest.json',manifest);started=time.perf_counter()
    def run(e):
        check_frozen(protocol)
        with (args.output_dir/(e['name']+'.log')).open('w') as log:
            subprocess.run(['/usr/bin/time','-p',sys.executable,'scripts/train.py','--config',e['config'],'--output-dir',str((args.output_dir/e['name']).resolve())],stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((args.output_dir/e['name']/'summary.json').read_text());assert result['status']=='complete' and result['transitions']==1024000 and result['updates']==1000
        return e['name']
    try:
        with ThreadPoolExecutor(max_workers=5) as pool:
            for f in as_completed([pool.submit(run,e) for e in protocol['runs']]):
                name=f.result();manifest['completed'].append(name);manifest['wall_seconds']=time.perf_counter()-started;write_json(args.output_dir/'manifest.json',manifest);print('Completed',name,flush=True)
    except Exception as error:
        manifest.update(status='failed',error=repr(error));write_json(args.output_dir/'manifest.json',manifest);raise
    check_frozen(protocol);manifest.update(status='complete',wall_seconds=time.perf_counter()-started);write_json(args.output_dir/'manifest.json',manifest)


if __name__=='__main__':main()
