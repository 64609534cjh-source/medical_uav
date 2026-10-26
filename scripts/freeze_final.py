"""Write the final Test protocol only after finite tuning and engineering checks."""

import hashlib,json
from pathlib import Path
from datetime import datetime,timezone


def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    gate=read('results/m5_development_gate.json')
    method='s1' if gate['gate_pass'] else 'm4'
    if method=='s1':
        tuning=read('results/m5_tuning.json');assert tuning['seeds']==[1,2,3,4,5]
        assert len(tuning['audits'])==5 and all(a['first_half_model_and_sampler_exactly_m4'] for a in tuning['audits'])
    benchmark=read('results/m5_inference.json')
    assert benchmark['validated_days']>=1200
    assert all(v['all_actions_and_day_fields_exact'] for v in benchmark['validated_models'])
    pre=read('results/m5_preflight.json')
    for path,h in pre['frozen_results'].items():assert digest(path)==h,path
    assert digest('results/m3_mlp_ppo.json')==pre['frozen_m3_result_sha256']
    roots={'mlp':{s:Path(f'outputs/m3_mixed_mlp_ppo/seed_{s}') for s in range(1,6)},
           'm4':{s:Path(f'outputs/m4_structured_ppo/seed_{s}') for s in range(1,6)}}
    if method=='s1':roots['s1']={s:Path(f'outputs/m5_s1_{"development_123" if s<=3 else "extension_45"}/seed_{s}') for s in range(1,6)}
    entries=[]
    for name,models in roots.items():
        for seed,root in models.items():
            checkpoint=root/'best.pt'
            entries.append({'name':f'{name}_s{seed}','method':name,'learner_seed':seed,'selection':'preselected best on Dev-main','checkpoint':str(checkpoint),'checkpoint_sha256':digest(checkpoint),'policies':['ppo']})
    entries.append({'name':'fixed_rules','method':'rules','learner_seed':0,'policies':['random','edf','slack']})
    files=sorted(str(p) for p in Path('medical_uav').rglob('*.py'))+['scripts/evaluate.py','scripts/evaluate_suite.py','scripts/analyze_final.py','environment.yml','requirements-lock.txt']
    comparisons=[{'left':'m4','right':'mlp'},{'left':method,'right':'slack'}]
    if method=='s1':comparisons += [{'left':'s1','right':'m4'},{'left':'s1','right':'mlp'}]
    protocol={'milestone':'Frozen final demand-day Test','frozen_utc':datetime.now(timezone.utc).isoformat(),'final_candidates_frozen':True,
              'split':'test','demand_seed_range':[200000,200099],'loads':['low','medium','high'],'map_seed':0,'jobs':5,'diagnostics':True,
              'final_method':method,'entries':entries,'primary_comparisons':comparisons,
              'selection':'one original Dev-main best checkpoint per learner seed, equal-load daily on-time mean then mean daily reward, earliest exact tie; no per-load checkpoint or Test seed selection',
              'statistics':{'primary':'equal-load mean daily on_time_rate, averaged over five learners','secondary':['reward','per-load/type service','delivery/expiry counts','energy and transport'],
                            'learner_sample_sd_ddof':1,'bootstrap_repetitions':10000,'bootstrap_seed':20261026,'confidence_level':.95,
                            'bootstrap':'paired crossed learner-ID and demand-day-ID product bootstrap; same day draws across all loads and methods, common learner-ID pairing across methods; baselines fixed learner dimension 1',
                            'report_intervals':['joint percentile','conditional learner-only percentile','conditional demand-day-only percentile'],
                            'limitations':'only five learner repetitions; fixed map; not 1500 independent training runs'},
              'random_protocol':'uniform legal actions, per-day isolated SeedSequence([0,day,load_index,0])',
              'evaluation':'canonical original network forward and deterministic masked categorical argmax; optional inference optimization not used for scientific Test',
              'frozen_files':{p:digest(p) for p in files},
              'preconditions':{'gate_sha256':digest('results/m5_development_gate.json'),'inference_equivalence_sha256':digest('results/m5_inference.json'),'initial_development_diagnostics_sha256':digest('results/m5_initial_diagnostics.json'),'m4_results_unchanged':True},
              'post_test_policy':'No further model/selection/hyperparameter changes. Any later tuning is new exploration and this Test is consumed.'}
    path=Path('configs/final_test_protocol.json')
    if path.exists():raise ValueError('final protocol already exists; do not silently replace a frozen Test design')
    path.write_text(json.dumps(protocol,indent=2)+'\n')
    print('Frozen final method:',method,'; models:',len(entries)-1,'; Test not yet evaluated.')


if __name__=='__main__':main()
