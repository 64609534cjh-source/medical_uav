"""Seal all fifteen candidates and analysis code before new Test-v3."""
import json,hashlib,subprocess
from pathlib import Path
from datetime import datetime,timezone
from medical_uav.training.trainer import load_checkpoint
from medical_uav.training.io import write_json
from run_gen3 import check_frozen


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    path=Path('configs/gen3_test_v3_protocol.json');assert not path.exists()
    p=json.loads(Path('configs/gen3_protocol.json').read_text());check_frozen(p)
    audit=json.loads(Path('results/gen3_training_audit.json').read_text());assert audit['all_checks_passed'] and len(audit['runs'])==10
    assert json.loads(Path('outputs/gen3_architectures/manifest.json').read_text())['status']=='complete'
    entries=json.loads(Path('configs/gen3_holdout_freeze.json').read_text())['reference_models'][:]
    for m in ['urgency','coordinated']:
        for s in range(1,6):
            cp=f'outputs/gen3_architectures/{m}_s{s}/best.pt';state=load_checkpoint(cp)
            es=[json.loads(x) for x in (Path(cp).parent/'evaluations.jsonl').read_text().splitlines()];best=max(es,key=lambda r:(r['policies']['ppo']['mean_on_time_rate'],r['policies']['ppo']['mean_reward']));assert state['steps']==best['transitions']
            entries.append({'name':f'{m}_s{s}','method':m,'learner_seed':s,'checkpoint':cp,'checkpoint_sha256':sha(cp),'policies':['ppo'],'selection':'best on Dev-main only; all learners retained'})
    for e in entries:assert sha(e['checkpoint'])==e['checkpoint_sha256']
    entries.append({'name':'baselines','policies':['slack','edf','random'],'learner_seed':0})
    files=list(p['frozen_files'])+['configs/gen3_protocol.json','scripts/analyze_gen3.py','scripts/freeze_gen3_final.py','scripts/analyze_gen2.py','scripts/analyze_gen3_initial.py','scripts/inference_gen3.py','configs/gen3_inference_validation_protocol.json','medical_uav/evaluation/final_stats.py','results/gen3_initial_diagnostics.json','results/gen3_training_audit.json','results/gen3_development.json','results/gen3_inference.json']
    verification=json.loads(Path('configs/gen3_inference_validation_protocol.json').read_text())
    for e in verification['entries']:
        root='outputs/gen3_reference_inference_validation' if e['method']=='destination' else 'outputs/gen3_new_inference_validation'
        r=json.loads(Path(root+'/'+e['name']+'.json').read_text())
        assert all(v['all_actions_and_complete_day_fields_exact'] for v in r['modes'].values())
    write_json(path,{'generation':3,'frozen_utc':datetime.now(timezone.utc).isoformat(),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'final_candidates_frozen':True,'split':'test-v3','demand_seed_range':[400000,400099],'loads':['low','medium','high'],'map_seed':0,'jobs':5,'diagnostics':True,'failure_diagnostics':True,'entries':entries,'statistics':p['statistics'],'primary_comparisons':p['primary_comparisons'],'scope':'all predeclared methods and learners irrespective of Dev ranking; canonical original forwards, no fast policy in scientific final evaluation','frozen_files':{f:sha(f) for f in sorted(set(files))}})
    print('Frozen 15 models and Test-v3 statistics;20 models passed full Dev inference equivalence in two optional modes.')


if __name__=='__main__':main()
