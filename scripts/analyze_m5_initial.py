"""Read M5 pre-tuning diagnostics without using Test or changing checkpoints."""
import json
from pathlib import Path
from statistics import mean,stdev
import numpy as np
ROOT=Path.cwd()


def load(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(l) for l in Path(path).read_text().splitlines()]


def summarize(records):
 out={}
 for load_name in ('low','medium','high'):
  rs=[r for r in records if r['episode_load']==load_name]
  out[load_name]={k:mean(r[k] for r in rs) for k in ('on_time_rate','reward','on_time','late_deliveries','expired','km_flown','charge_minutes','wait_minutes')}
  d=[r['diagnostics'] for r in rs]
  out[load_name]['outcome_totals']={k:sum(x['outcomes'][k] for x in d) for k in d[0]['outcomes']}
  out[load_name]['type']={}
  for kind in d[0]['outcomes_by_type']:
   ts=[x['outcomes_by_type'][kind] for x in d];out[load_name]['type'][kind]={'counts':{k:sum(t[k] for t in ts) for k in ts[0]},'daily_rates':{k:mean(t[k]/t['released'] for t in ts if t['released']) for k in ('on_time','expired_unpicked','expired_after_pick')}}
  delivery={k:sum(x['delivery'][k] for x in d) for k in d[0]['delivery']}
  out[load_name]['delivery']={**delivery,'boxes_per_effective_delivery':delivery['delivered_boxes']/delivery['effective_actions'] if delivery['effective_actions'] else None,'batch_fraction':delivery['batch_actions']/delivery['effective_actions'] if delivery['effective_actions'] else None}
  out[load_name]['resource_histograms']={}
  for key in d[0]['resources']:
   bins=d[0]['resources'][key]['edges'];counts=np.sum([x['resources'][key]['counts'] for x in d],axis=0).tolist();n=sum(x['resources'][key]['n'] for x in d)
   out[load_name]['resource_histograms'][key]={'edges':bins,'counts':counts,'n':n,'event_weighted_mean':sum(x['resources'][key]['sum'] for x in d)/n if n else None}
 out['equal']=mean(out[l]['on_time_rate'] for l in ('low','medium','high'))
 return out


def main():
 extra=Path('outputs/m5_dev_extra_initial');main_root=Path('outputs/m5_main_diagnostics')
 assert load(extra/'manifest.json')['status']==load(main_root/'manifest.json')['status']=='complete'
 result={'scope':'M5 development diagnostics before tuning; not Test','per_model':{},'pooled_equal_seed_weight':{},'main_reference_verified':True}
 for split,root in [('dev-extra',extra),('dev-main',main_root)]:
  groups={}
  for e in load(root/'manifest.json')['protocol']['entries']:
   rs=load(root/(e['name']+'.json'))['episodes'];assert len(rs)==(300 if split=='dev-extra' else 120)
   result['per_model'][split+'_'+e['name']]=summarize(rs)
   groups.setdefault(e['method']+'_'+e.get('selection','rule'),[]).extend(rs)
   if split=='dev-main' and e['method']=='m4':
    old=rows(f"outputs/m4_structured_ppo/seed_{e['learner_seed']}/evaluation_episodes.jsonl")
    candidates=[r for r in load('results/m4_structured_audit.json')[e['selection']+'_per_seed'] if r['learner_seed']==e['learner_seed']];step=candidates[0]['transitions']
    index={(r['episode_load'],r['demand_seed']):r for r in old if r['policy']=='ppo' and r['transitions']==step}
    for r in rs:
     a=index[(r['episode_load'],r['demand_seed'])]
     assert all(r[k]==a[k] for k in r if k not in ('diagnostics',))
  result['pooled_equal_seed_weight'][split]={name:summarize(rs) for name,rs in groups.items()}
 result['updates']=load('outputs/m5_audit/m4_update_diagnostics.json')
 Path('results/m5_initial_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['# M5 pre-tuning diagnostics','', 'Dev-extra 100040–100139 was absent from all 152 recorded local output files before this stage. It is explicitly a development set. Test remains sealed. Original Dev-main selected checkpoints are used unchanged across all loads; no per-load reselection.','', '| Split / model | Low % | Medium % | High % | Equal % |','|---|---:|---:|---:|---:|']
 for split,groups in result['pooled_equal_seed_weight'].items():
  for name,r in groups.items():lines.append('| '+split+' / '+name+' | '+' | '.join(f"{100*r[l]['on_time_rate']:.2f}" for l in ('low','medium','high'))+f" | {100*r['equal']:.2f} |")
 lines+=['','## Matched best-to-last changes on Dev-extra','','| Seed | M4 best % | M4 last % | Last − best (pp) | MLP best % | M4 − MLP best (pp) |','|---:|---:|---:|---:|---:|---:|']
 for seed in range(1,6):
  b=result['per_model'][f'dev-extra_m4_best_s{seed}']['equal'];l=result['per_model'][f'dev-extra_m4_last_s{seed}']['equal'];m=result['per_model'][f'dev-extra_mlp_best_s{seed}']['equal'];lines.append(f'| {seed} | {100*b:.2f} | {100*l:.2f} | {100*(l-b):+.2f} | {100*m:.2f} | {100*(b-m):+.2f} |')
 lines+=['','## Service, failure and transport','','Counts pool equal-size day sets across the same five seeds; they retain numerators/denominators rather than averaging conditional ratios. Resource histograms use event weighting and are preserved in JSON.','','| Split / load / M4 selection | Reward/day | On-time count | Late count | Unpicked expiry | Picked expiry / pickups | Boxes/effective drop | Batch % | Km/day | Charge min/day | Wait min/day |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for split in ('dev-main','dev-extra'):
  for load_name in ('low','medium','high'):
   for selection in ('best','last'):
    r=result['pooled_equal_seed_weight'][split]['m4_'+selection][load_name];o=r['outcome_totals'];d=r['delivery'];lines.append(f"| {split} / {load_name} / {selection} | {r['reward']:.3f} | {o['on_time']} | {o['late']} | {o['expired_unpicked']} | {o['expired_after_pick']} / {o['picked']} | {d['boxes_per_effective_delivery']:.3f} | {100*d['batch_fraction']:.2f} | {r['km_flown']:.2f} | {r['charge_minutes']:.2f} | {r['wait_minutes']:.2f} |")
 lines+=['','Per-seed/load/type released counts, daily on-time/expiry rates, both expiry failure modes, effective-drop denominators and battery/payload histograms are in the JSON. All new Dev-main scores exactly match the saved M4 records.','', 'M4 LR is constant 3e-4. Target KL 0.02 actually stops remaining joint Actor/Critic updates. Across 25 late checkpoint intervals, the descriptive correlation of interval mean sampled-action final KL and score change is '+f"{result['updates']['late_interval_kl_vs_score_change_pearson']:.3f}"+'. This does not establish causality, and does not show the hypothesized negative association. The largest-decline seed is not the one with the largest late KL. M4 did not save full old action distributions, so exact historical KL cannot be recovered.','', 'The primary S1 experiment will alter only the predeclared LR schedule, while adding exact categorical KL and normalized entropy diagnostics without changing KL stopping. There is currently no evidence to prioritize a stricter-KL S2 run. No claim is made that KL can never matter.','']
 Path('results/m5_initial_diagnostics.md').write_text('\n'.join(lines));print('\n'.join(lines[:23]))

if __name__=='__main__':main()
