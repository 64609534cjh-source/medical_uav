"""Audit fixed-budget S1 and apply the preregistered development-only gate."""
import argparse
from collections import Counter,defaultdict
import hashlib,json,math
from pathlib import Path
from statistics import mean,stdev
import sys,tarfile
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1] if Path(__file__).parent.name=='scripts' else Path.cwd()
sys.path.insert(0,str(ROOT))
from scripts.analyze_m3 import day_index,summarize_days,finite_tensors
from scripts.analyze_m5_initial import summarize
from medical_uav.training.trainer import load_checkpoint,TrainConfig,learning_rate_at
from medical_uav.training.io import write_json

LOADS=('low','medium','high')

def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(l) for l in Path(p).read_text().splitlines()]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run_for(seed):return ROOT/f'outputs/m5_s1_{"development_123" if seed<=3 else "extension_45"}/seed_{seed}'

def audit(seed,protocol):
 run=run_for(seed);meta=read(run/'metadata.json');config=read(run/'config.json');summary=read(run/'summary.json')
 assert not meta['git_dirty'] and meta['resume_mode']=='fresh' and meta['resume_from'] is None
 assert config==read(ROOT/protocol['configurations'][str(seed)])
 assert summary['transitions']==1024000 and summary['updates']==1000 and summary['status']=='complete'
 manifest=read(run/'source_manifest.json')
 with tarfile.open(run/'source.tar.gz') as archive:
  for p,h in protocol['frozen_file_sha256'].items():
   assert manifest[p]==h and hashlib.sha256(archive.extractfile(p).read()).hexdigest()==h
 ev=rows(run/'evaluations.jsonl');all_days=rows(run/'evaluation_episodes.jsonl')
 assert [r['transitions'] for r in ev]==list(range(0,1024001,102400)) and len(all_days)==1680
 checkpoints=[]
 for e in ev:
  cp=load_checkpoint(run/e['checkpoint']);assert cp['steps']==e['transitions'] and cp['source_commit']==meta['git_commit']
  finite_tensors(cp['model_state_dict']);finite_tensors(cp['optimizer_state_dict'])
  days=[r for r in all_days if r['policy']=='ppo' and r['transitions']==e['transitions']];day_index(days)
  s=summarize_days(days);assert math.isclose(s['mean_on_time_rate'],e['policies']['ppo']['mean_on_time_rate'],abs_tol=1e-12)
  checkpoints.append({'steps':e['transitions'],'checkpoint_sha256':digest(run/e['checkpoint']),**s})
 selected=max(checkpoints,key=lambda r:(r['mean_on_time_rate'],r['mean_reward']))
 best=load_checkpoint(run/'best.pt');last=load_checkpoint(run/'last.pt');assert best['steps']==selected['steps'] and last['steps']==1024000
 for cp,name in ((best,f"checkpoint_{best['steps']:09d}.pt"),(last,'checkpoint_001024000.pt')):
  other=load_checkpoint(run/name);assert all(torch.equal(v,other['model_state_dict'][k]) for k,v in cp['model_state_dict'].items())
 previous=load_checkpoint(ROOT/f'outputs/m4_structured_ppo/seed_{seed}/checkpoint_000512000.pt')
 midpoint=load_checkpoint(run/'checkpoint_000512000.pt')
 assert all(torch.equal(v,previous['model_state_dict'][k]) for k,v in midpoint['model_state_dict'].items()),'S1 early half diverged from S0'
 assert midpoint['collector']==previous['collector']
 updates=rows(run/'updates.jsonl');assert len(updates)==1000
 cfg=TrainConfig.from_dict(config)
 for i,u in enumerate(updates,1):
  assert u['update_index']==i and u['transitions']==1024*i and u['invalid_actions']==0
  assert u['learning_rate']==learning_rate_at(cfg,1024*i)
  for k in ('value_loss','entropy','final_approx_kl','exact_categorical_kl_mean','exact_categorical_kl_choice_mean','old_normalized_entropy_choice_mean','gradient_norm','return_std','advantage_std'):
   assert isinstance(u[k],(float,int)) and math.isfinite(u[k])
  assert u['initial_ratio_max_error']<1e-5 and u['old_distribution_float32_bytes']==106496
 episodes=rows(run/'episodes.jsonl');counts=Counter(r['episode_load'] for r in episodes)
 assert all(r['complete_day'] and r['invalid_actions']==0 and 0<=r['demand_seed']<100000 for r in episodes)
 assert counts==summary['episodes_by_load']
 assert sum(summary['transitions_by_load'].values())==1024000
 return {'seed':seed,'source_commit':meta['git_commit'],'source_manifest_sha256':digest(run/'source_manifest.json'),'updates':1000,
         'complete_training_days':len(episodes),'checkpoints':checkpoints,'best':selected,'last':checkpoints[-1],
         'tail':mean(r['mean_on_time_rate'] for r in checkpoints[-3:]),'first_half_model_and_sampler_exactly_m4':True,
         'timings':{k:summary[k] for k in ('wall_seconds','training_seconds','eval_seconds')},
         'kl':{k:{'mean':mean(u[k] for u in updates),'late_mean':mean(u[k] for u in updates[500:]),'max':max(u[k] for u in updates)} for k in ('final_approx_kl','exact_categorical_kl_mean','exact_categorical_kl_choice_mean','old_normalized_entropy_choice_mean')},
         'early_stops':sum(u['kl_early_stop'] for u in updates),'optimizer_steps':sum(u['optimizer_steps'] for u in updates),
         'last_lr':updates[-1]['learning_rate'],'mix':summary['transitions_by_load']}


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--seeds',type=int,nargs='+',default=[1,2,3]);parser.add_argument('--extra-dir',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();torch.set_num_threads(1)
 protocol=read(ROOT/'configs/m5_protocol.json');audits=[audit(s,protocol) for s in args.seeds];old=read(ROOT/'results/m4_structured_audit.json')
 assert len({a['source_commit'] for a in audits})==1
 selection={};extra={};old_extra=read(ROOT/'results/m5_initial_diagnostics.json');new_extra_days=[]
 assert read(args.extra_dir/'manifest.json')['status']=='complete'
 for selection_name in ('best','last'):
  extra[selection_name]={}
  for seed in args.seeds:
   value=read(args.extra_dir/f's1_{selection_name}_s{seed}.json');rs=value['episodes'];assert len(rs)==300 and all(r['split']=='dev-extra' for r in rs)
   extra[selection_name][str(seed)]=summarize(rs)
   if selection_name=='best':new_extra_days.extend(rs)
 main_deltas={}
 for metric in ('best','last','tail'):
  a=[r[metric]['mean_on_time_rate'] if metric!='tail' else r[metric] for r in audits]
  b=[]
  for seed in args.seeds:
   if metric=='tail':b.append(mean(r['mean_on_time_rate'] for r in old['curves'][str(seed)][-3:]))
   else:b.append(next(r['mean_on_time_rate'] for r in old[metric+'_per_seed'] if r['learner_seed']==seed))
  deltas=[n-o for n,o in zip(a,b)]
  main_deltas[metric]={'s1':a,'s0':b,'deltas':deltas,'mean_delta':mean(deltas),'sample_sd_delta':stdev(deltas) if len(deltas)>1 else None}
 extra_deltas={}
 for selection_name in ('best','last'):
  vals=[extra[selection_name][str(seed)]['equal']-old_extra['per_model'][f'dev-extra_m4_{selection_name}_s{seed}']['equal'] for seed in args.seeds]
  extra_deltas[selection_name]={'deltas':vals,'mean_delta':mean(vals),'sample_sd_delta':stdev(vals)}
 best_pooled=summarize(new_extra_days);old_pooled={load:{'rate':mean(old_extra['per_model'][f'dev-extra_m4_best_s{s}'][load]['on_time_rate'] for s in args.seeds),
                    'types':{kind:mean(old_extra['per_model'][f'dev-extra_m4_best_s{s}'][load]['type'][kind]['daily_rates']['on_time'] for s in args.seeds) for kind in best_pooled[load]['type']}} for load in LOADS}
 load_delta={l:best_pooled[l]['on_time_rate']-old_pooled[l]['rate'] for l in LOADS}
 type_delta={l:{k:best_pooled[l]['type'][k]['daily_rates']['on_time']-old_pooled[l]['types'][k] for k in best_pooled[l]['type']} for l in LOADS}
 g=protocol['extension_gate'];checks={'best_main':main_deltas['best']['mean_delta']>=g['best_main_delta_at_least'],
 'last_main':main_deltas['last']['mean_delta']>=g['last_main_delta_at_least'],'tail_main':main_deltas['tail']['mean_delta']>=g['tail_main_delta_at_least'],
 'best_extra':extra_deltas['best']['mean_delta']>=g['best_extra_delta_at_least'],'last_extra':extra_deltas['last']['mean_delta']>=g['last_extra_delta_at_least'],
 'per_load_extra':min(load_delta.values())>=g['per_load_best_extra_delta_at_least'],
 'per_type_extra':min(v for d in type_delta.values() for v in d.values())>=g['per_type_best_extra_delta_at_least']}
 result={'scope':'M5 S1 development comparison; Test unused','seeds':args.seeds,'audits':audits,'main_deltas':main_deltas,'extra_deltas':extra_deltas,
         'extra_best_load_deltas':load_delta,'extra_best_type_deltas':type_delta,'extra':extra,'gate_checks':checks,'gate_pass':all(checks.values()),
         'gate_is_practical_not_statistical':True,'gate_applies_to_this_seed_set':args.seeds==[1,2,3],'source_commit':audits[0]['source_commit']}
 write_json(args.output,result);print(json.dumps({k:result[k] for k in ('main_deltas','extra_deltas','gate_checks','gate_pass')},indent=2))

if __name__=='__main__':main()
