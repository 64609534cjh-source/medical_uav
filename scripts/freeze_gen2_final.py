"""Seal all fifteen selected-best models before the new Test-v2 is opened."""
import hashlib,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
from medical_uav.training.io import write_json
from medical_uav.training.trainer import load_checkpoint


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    path=Path('configs/gen2_test_v2_protocol.json');assert not path.exists(),'never overwrite a frozen protocol'
    freeze=json.loads(Path('configs/gen2_holdout_freeze.json').read_text());p=json.loads(Path('configs/gen2_protocol.json').read_text());audit=json.loads(Path('results/gen2_training_audit.json').read_text())
    assert audit['all_checks_passed'] and len(audit['runs'])==10
    assert json.loads(Path('outputs/gen2_architectures/manifest.json').read_text())['status']=='complete'
    for f,h in p['frozen_files'].items():assert sha(f)==h,f
    for f,h in freeze['frozen_generation1_files'].items():assert sha(f)==h,f
    entries=freeze['reference_models'][:]
    for m in ['destination','relational']:
        for s in range(1,6):
            cp=f'outputs/gen2_architectures/{m}_s{s}/best.pt';state=load_checkpoint(cp)
            records=[json.loads(x) for x in Path(cp).parent.joinpath('evaluations.jsonl').read_text().splitlines()]
            best=max(records,key=lambda r:(r['policies']['ppo']['mean_on_time_rate'],r['policies']['ppo']['mean_reward']))
            assert state['steps']==best['transitions']
            entries.append({'name':f'{m}_s{s}','method':m,'learner_seed':s,'checkpoint':cp,'checkpoint_sha256':sha(cp),'policies':['ppo'],'selection':'preselected best on Dev-main only'})
    entries.append({'name':'baselines','policies':['slack','edf','random'],'learner_seed':0})
    for e in entries:
        if 'checkpoint' in e:assert sha(e['checkpoint'])==e['checkpoint_sha256']
    files=list(p['frozen_files'])+['configs/gen2_protocol.json','scripts/analyze_gen2.py','scripts/freeze_gen2_final.py','medical_uav/evaluation/final_stats.py','results/gen2_training_audit.json','results/gen2_development.json','results/gen2_latency.json']
    result={'generation':2,'frozen_utc':datetime.now(timezone.utc).isoformat(),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'split':'test-v2','final_candidates_frozen':True,'demand_seed_range':[300000,300099],'loads':['low','medium','high'],'map_seed':0,'jobs':5,'diagnostics':True,'entries':entries,'statistics':p['statistics'],'comparisons':p['comparisons'],'scope':'all three predeclared methods regardless of development ranking; no model or seed selected on Test-v2','frozen_files':{f:sha(f) for f in sorted(set(files))}}
    write_json(path,result);print('Frozen 15 selected-best models, 100 new Test-v2 demand days, statistical code and development evidence.')


if __name__=='__main__':main()
