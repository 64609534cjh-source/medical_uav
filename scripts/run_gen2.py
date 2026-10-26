"""Ten preregistered scratch runs; no performance-based stopping or retuning."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib,json,subprocess,sys,time
from pathlib import Path
from medical_uav.training.io import write_json


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_frozen(protocol):
    for path,sha in protocol['frozen_files'].items():assert digest(path)==sha,path
    first=json.loads(Path('configs/gen2_holdout_freeze.json').read_text())
    for path,sha in first['frozen_generation1_files'].items():assert digest(path)==sha,path
    for e in first['reference_models']:assert digest(e['checkpoint'])==e['checkpoint_sha256']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,default=Path('configs/gen2_protocol.json'));p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    protocol=json.loads(args.protocol.read_text());check_frozen(protocol)
    assert not args.output_dir.exists(),'fresh output required';args.output_dir.mkdir(parents=True)
    manifest={'protocol':protocol,'protocol_sha256':digest(args.protocol),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'status':'running','completed':[]}
    write_json(args.output_dir/'manifest.json',manifest);started=time.perf_counter()
    def run(e):
        check_frozen(protocol)
        with (args.output_dir/(e['name']+'.log')).open('w') as log:
            subprocess.run(['/usr/bin/time','-p',sys.executable,'scripts/train.py','--config',e['config'],'--output-dir',str((args.output_dir/e['name']).resolve())],stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((args.output_dir/e['name']/'summary.json').read_text());assert result['transitions']==1024000 and result['updates']==1000 and result['status']=='complete'
        return e['name']
    try:
        with ThreadPoolExecutor(max_workers=5) as pool:
            for f in as_completed([pool.submit(run,e) for e in protocol['runs']]):
                name=f.result();manifest['completed'].append(name);manifest['wall_seconds']=time.perf_counter()-started
                write_json(args.output_dir/'manifest.json',manifest);print('Completed',name,flush=True)
    except Exception as exc:
        manifest.update(status='failed',error=repr(exc));write_json(args.output_dir/'manifest.json',manifest);raise
    check_frozen(protocol);manifest.update(status='complete',wall_seconds=time.perf_counter()-started);write_json(args.output_dir/'manifest.json',manifest)


if __name__=='__main__':main()
