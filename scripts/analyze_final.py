"""Analyze only the frozen one-time Test suite, preserving learner/day pairing."""

import argparse
import hashlib,json,math
from pathlib import Path
from statistics import mean,stdev
import sys
ROOT=Path(__file__).resolve().parents[1] if Path(__file__).parent.name=='scripts' else Path.cwd()
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from medical_uav.evaluation.final_stats import crossed_bootstrap
from medical_uav.training.io import write_json
from scripts.analyze_m5_initial import summarize

LOADS=('low','medium','high')
DAYS=list(range(200000,200100))


def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def matrix(records):
    index={(r['episode_load'],r['demand_seed']):r for r in records}
    assert len(index)==len(records)==300 and set(index)=={(l,d) for l in LOADS for d in DAYS}
    for r in records:
        assert r['split']=='test' and r['map_seed']==0 and r['complete_day'] and r['invalid_actions']==0
        assert r['released']==r['delivered']+r['expired']
        assert r['on_time_rate']==(r['on_time']/r['released'] if r['released'] else 0)
    return np.array([[index[(l,d)]['on_time_rate'] for d in DAYS] for l in LOADS])


def draw(result,path):
    comparisons=result['comparisons']; names=list(comparisons)
    fig,axes=plt.subplots(1,4,figsize=(13,4.5),layout='constrained')
    for ax,load in zip(axes,(*LOADS,'equal')):
        for j,name in enumerate(names):
            r=comparisons[name]['loads'][load];value=100*r['estimate'];a,b=[100*v for v in r['joint_95_percentile']]
            ax.plot([a,b],[j,j],color='#225E9A',lw=2)
            ax.scatter([value],[j],color='#225E9A',s=25,zorder=3)
        ax.axvline(0,color='#555555',lw=1,ls=':');ax.grid(axis='x',alpha=.2)
        ax.set_yticks(range(len(names)),labels=[n.replace('_minus_',' − ').upper() for n in names])
        ax.invert_yaxis();ax.set(title=load.title(),xlabel='On-time rate difference (pp)')
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Frozen Test: paired differences and 95% crossed-bootstrap intervals\nFive learners × 100 shared demand-day IDs; loads resampled together',fontsize=13)
    for ext in ('png','svg'):fig.savefig(str(path)+'_paired.'+ext,dpi=180)
    plt.close(fig)


def report(result,path):
    p=result['protocol'];final=p['final_method'];names={'mlp':'Flat MLP','m4':'Structured S0','s1':'Structured S1','slack':'Slack-aware','edf':'EDF','random':'Random'}
    lines=['# Frozen final Test comparison','', 'All candidate checkpoints and this analysis protocol were frozen before opening Test. Every learned method uses five independently trained learners and one preselected Dev-main best checkpoint per learner, shared across all loads. No seed or model is selected using Test.','',
           '## Main result','', '| Method | Low % | Medium % | High % | Equal-load % | Joint 95% interval, equal % |','|---|---:|---:|---:|---:|---:|']
    for name in (*[m for m in ('mlp','m4','s1') if m in result['performance']], 'slack','edf','random'):
        r=result['performance'][name]['loads'];learned=name in ('mlp','m4','s1')
        vals=[f"{100*r[l]['estimate']:.2f}"+(f" ± {100*r[l]['learner_sample_sd']:.2f}" if learned else '') for l in (*LOADS,'equal')]
        ci=r['equal']['joint_95_percentile'];lines.append('| '+names[name]+' | '+' | '.join(vals)+f" | [{100*ci[0]:.2f}, {100*ci[1]:.2f}] |")
    lines+=['','± is sample SD across five learner means, not a confidence interval. Interval estimates use the separately predeclared paired learner/day bootstrap. The final selected structural method is **'+names[final]+'**.','', '## All learner repetitions','', '| Method / seed | Low % | Medium % | High % | Equal % |','|---|---:|---:|---:|---:|']
    for name,seed_rows in result['per_seed'].items():
        for seed,values in seed_rows.items():lines.append('| '+names[name]+f' / {seed} | '+' | '.join(f'{100*v:.2f}' for v in values)+' |')
    lines+=['','## Paired primary comparisons','', '| Comparison / load | Difference (pp) | Learner SD (pp) | Joint 95% interval (pp) |','|---|---:|---:|---:|']
    for name,value in result['comparisons'].items():
        for load,r in value['loads'].items():
            ci=r['joint_95_percentile'];lines.append(f"| {name} / {load} | {100*r['estimate']:+.2f} | {100*r['learner_sample_sd']:.2f} | [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}] |")
    lines+=['','The JSON also retains learner-only and day-only intervals. A common demand-day resample is used across loads and compared policies. The learner-ID resample preserves the declared seed pairings; identical trajectories are not assumed. No calculation treats 1,500 model/load/day records as 1,500 independent training runs. With only five training repetitions, the bootstrap is an uncertainty estimate with limited empirical support, not a guarantee.','', '## Service outcomes on final Test','', '| Load / method | Reward/day | On-time count | Late count | Unpicked expiry | Picked expiry / pickups | Boxes/effective drop | Flight km/day |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for load in LOADS:
        for name in ('mlp',final,'slack'):
            r=result['service'][name][load];o=r['outcome_totals'];d=r['delivery'];lines.append(f"| {load} / {names[name]} | {r['reward']:.3f} | {o['on_time']} | {o['late']} | {o['expired_unpicked']} | {o['expired_after_pick']} / {o['picked']} | {d['boxes_per_effective_delivery']:.3f} | {r['km_flown']:.2f} |")
    lines+=['','Counts sum five equal-size Test runs for learned methods and one run for the deterministic rule; these totals have different repetition counts and are shown as explicit outcome denominators, not directly comparable event volumes. Day-level means and rates are the comparison metrics.','', '| Load / type | MLP on time % | Final on time % | Slack on time % |','|---|---:|---:|---:|']
    for load in LOADS:
        for kind in result['service'][final][load]['type']:
            vals=[result['service'][n][load]['type'][kind]['daily_rates']['on_time'] for n in ('mlp',final,'slack')]
            lines.append(f'| {load} / {kind} | '+' | '.join(f'{100*v:.2f}' for v in vals)+' |')
    lines+=['','Type-specific rates average nonempty-type demand days; actual released/picked/outcome counts, conditional post-pick expiry, charging/waiting and resource histograms are retained in JSON. None of these Test subgroups is used for subsequent tuning.','', '## Reproducibility and scope','',
           f"Test suite source commit: `{result['source_commit']}`. Protocol SHA256: `{result['protocol_sha256']}`.",
           'Test demand-day IDs: 200000–200099, Low/Medium/High, fixed map seed 0. Random is uniform over legal actions with the frozen isolated SeedSequence([0,day,load_index,0]) stream. Slack and EDF were rerun on these Test days; their Dev scores were not reused.',
           f"There are {result['coverage']['episodes']} complete Test episodes, {result['coverage']['decisions']} decisions, zero illegal actions and no incomplete day. Every source/model hash matches the frozen protocol.",
           'The tested method uses the canonical training-network forward and the same deterministic masked categorical argmax evaluation as M3/M4. The optional verified inference optimization is reported separately and does not redefine the scientific Test policy.',
           'This validates demand-day generalization under the fixed simulator/map. It does not establish generalization to new maps, real UAV operations or altered rewards.',
           'No model/selection/schedule changes follow these Test results. Further tuning would be a new exploration, and this Test would no longer be untouched.',
           '', 'Methodological context: [RL evaluation with few runs](https://arxiv.org/abs/2108.13264) motivates retaining training uncertainty; [model-selection bias](https://www.jmlr.org/papers/v11/cawley10a.html) motivates the final frozen holdout. The crossed learner/day bootstrap here is explicitly specified in the protocol; it is not claimed to be an exact implementation of the cited benchmark methodology.','']
    path.write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,default=ROOT/'outputs/final_test');parser.add_argument('--output',type=Path,default=ROOT/'results/final_test');args=parser.parse_args()
    manifest=read(args.run_dir/'manifest.json');protocol=manifest['protocol'];assert manifest['status']=='complete' and protocol['final_candidates_frozen']
    assert manifest['protocol_sha256']==digest(ROOT/'configs/final_test_protocol.json')
    arrays={};daily={};per_seed={};coverage={'episodes':0,'decisions':0,'invalid_actions':0}
    for entry in protocol['entries']:
        if entry.get('checkpoint'):assert digest(ROOT/entry['checkpoint'])==entry['checkpoint_sha256']
        document=read(args.run_dir/(entry['name']+'.json'));rs=document['episodes'];coverage['episodes']+=len(rs);coverage['decisions']+=sum(r['decisions'] for r in rs)
        for policy in entry['policies']:
            group=[r for r in rs if r['policy']==policy];a=matrix(group)
            name=entry['method'] if policy=='ppo' else policy
            arrays.setdefault(name,[]).append((entry.get('learner_seed',0),a));daily.setdefault(name,[]).extend(group)
    data={}
    for name,items in arrays.items():
        items.sort(key=lambda x:x[0]);data[name]=np.stack([a for _,a in items])
        if name in ('mlp','m4','s1'):
            assert [s for s,_ in items]==list(range(1,6))
            per_seed[name]={str(s):[*a.mean(axis=1).tolist(),float(a.mean())] for s,a in items}
    settings=protocol['statistics'];performance={name:crossed_bootstrap(a,np.zeros((1,3,100)),settings['bootstrap_repetitions'],settings['bootstrap_seed']) for name,a in data.items()}
    comparisons={};paired=[]
    for comparison in protocol['primary_comparisons']:
        left,right=comparison['left'],comparison['right'];name=left+'_minus_'+right
        comparisons[name]=crossed_bootstrap(data[left],data[right],settings['bootstrap_repetitions'],settings['bootstrap_seed'])
        delta=data[left]-data[right]
        for s in range(5):
            for l,load in enumerate(LOADS):
                for d,day in enumerate(DAYS):paired.append({'comparison':name,'learner_seed':s+1,'load':load,'demand_seed':day,'difference':float(delta[s,l,d])})
    result={'source_commit':manifest['source_commit'],'protocol':protocol,'protocol_sha256':manifest['protocol_sha256'],'performance':performance,'per_seed':per_seed,
            'comparisons':comparisons,'coverage':coverage,'service':{name:summarize(rs) for name,rs in daily.items()},'test_used_once_after_freeze':True}
    write_json(args.output.with_suffix('.json'),result);write_json(args.output.parent/(args.output.name+'_paired.json'),{'unit':'learner-ID x load x shared Test demand-day ID; descriptive rows','rows':paired})
    report(result,args.output.with_suffix('.md'));draw(result,args.output)
    print(json.dumps({name:r['loads']['equal'] for name,r in performance.items()},indent=2))

if __name__=='__main__':main()
