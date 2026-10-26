"""Gen3: source/weight audits and prespecified matched Dev / Test-v3 analysis."""
import argparse,hashlib,json,tarfile
from pathlib import Path
import numpy as np
from medical_uav.training.io import write_json
from medical_uav.training.trainer import load_checkpoint
from medical_uav.evaluation.final_stats import crossed_bootstrap
from medical_uav.evaluation.type_stats import type_bootstrap
from analyze_gen2 import data_array,describe
from run_gen3 import check_frozen

METHODS=('destination','urgency','coordinated');LOADS=('low','medium','high');TYPES=('plasma','medicine','sample','supplies')
COLORS={'destination':'#657084','urgency':'#2574a9','coordinated':'#ba7423'}
STYLES={'destination':'--','urgency':'-','coordinated':'-.'}


def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def baseline_runs():return {e['learner_seed']:Path(e['checkpoint']).parent for e in read('configs/gen3_holdout_freeze.json')['reference_models']}
def run_at(m,s):return baseline_runs()[s] if m=='destination' else Path(f'outputs/gen3_architectures/{m}_s{s}')


def audit_training():
    p=read('configs/gen3_protocol.json');check_frozen(p);assert read('outputs/gen3_architectures/manifest.json')['status']=='complete';audit=[]
    for entry in p['runs']:
        root=Path('outputs/gen3_architectures')/entry['name'];config=read(entry['config']);meta=read(root/'metadata.json');summary=read(root/'summary.json')
        assert meta['git_dirty'] is False and meta['resume_from'] is None and meta['resume_mode']=='fresh'
        assert meta['config']==config==read(root/'config.json') and summary['status']=='complete' and summary['transitions']==1024000 and summary['updates']==1000
        manifest=read(root/'source_manifest.json')
        with tarfile.open(root/'source.tar.gz') as archive:
            for path,h in manifest.items():assert hashlib.sha256(archive.extractfile(path).read()).hexdigest()==h
        for path,h in p['frozen_files'].items():
            if path.endswith('.py'):assert manifest[path]==h,(entry['name'],path)
        updates=lines(root/'updates.jsonl');evaluations=lines(root/'evaluations.jsonl');episodes=lines(root/'evaluation_episodes.jsonl');train_episodes=lines(root/'episodes.jsonl')
        assert len(updates)==1000 and [r['transitions'] for r in evaluations]==p['evaluation_transitions']
        for r in updates:
            assert r['invalid_actions']==0 and r['truncated_count']==0 and r['initial_ratio_max_error']<1e-5
            for k in ['policy_loss','value_loss','entropy','gradient_norm','initial_ratio_max_error','exact_categorical_kl_mean']:assert isinstance(r[k],(float,int)) and np.isfinite(r[k])
        assert all(r['complete_day'] and not r['truncated'] and not r['invalid_actions'] for r in episodes+train_episodes)
        chosen=max(evaluations,key=lambda r:(r['policies']['ppo']['mean_on_time_rate'],r['policies']['ppo']['mean_reward']))
        best=load_checkpoint(root/'best.pt');last=load_checkpoint(root/'last.pt');assert best['steps']==chosen['transitions'] and last['steps']==1024000
        for ev in evaluations:
            cp=load_checkpoint(root/ev['checkpoint']);assert cp['steps']==ev['transitions'];assert all(np.isfinite(v.numpy()).all() for v in cp['model_state_dict'].values())
            rs=[r for r in episodes if r['policy']=='ppo' and r['transitions']==ev['transitions']];assert len(rs)==120
            score=np.mean([np.mean([r['on_time_rate'] for r in rs if r['episode_load']==l]) for l in LOADS]);assert abs(score-ev['policies']['ppo']['mean_on_time_rate'])<1e-12
        audit.append({'name':entry['name'],'source_commit':meta['git_commit'],'best_steps':best['steps'],'evaluations':len(evaluations),'ppo_evaluation_days':1320,'updates':1000,'invalid_actions':0,'finite_required_training_diagnostics_and_checkpoint_parameters':True,'train_complete_days':len(train_episodes),'wall_seconds':summary['wall_seconds'],'training_seconds':summary['training_seconds'],'eval_seconds':summary['eval_seconds'],'transitions_by_load':summary['transitions_by_load'],'episodes_by_load':summary['episodes_by_load'],'best_sha256':sha(root/'best.pt'),'last_sha256':sha(root/'last.pt'),'source_manifest_sha256':sha(root/'source_manifest.json'),'tail_200_updates':{k:float(np.mean([r[k] for r in updates[-200:]])) for k in ['value_loss','entropy','exact_categorical_kl_mean','old_normalized_entropy_choice_mean','gradient_norm','kl_early_stop','optimizer_steps']}})
    return {'all_checks_passed':True,'generation':3,'budget_total':10240000,'formal_source_commits':sorted({r['source_commit'] for r in audit}),'prior_files_verified':len(read('configs/gen3_holdout_freeze.json')['protected_prior_files']),'runs':audit}


def summarize(rows,days):
    arrays={m:data_array(rows[m],days) for m in METHODS};result={'methods':{m:describe(a) for m,a in arrays.items()},'comparisons':{},'types':{},'failure_patterns':{}}
    baseline={(r['episode_load'],r['demand_seed']):{k:c['released'] for k,c in r['type_counts'].items()} for r in rows['destination'][0]}
    for m in METHODS:
        for rs in rows[m]:
            for r in rs:assert baseline[r['episode_load'],r['demand_seed']]=={k:c['released'] for k,c in r['type_counts'].items()}
        for metric in ['reward','served_rate','expiry_rate']:result['methods'][m][metric]=describe(data_array(rows[m],days,metric))
        rs=[r for group in rows[m] for r in group]
        if all('diagnostics' in r for r in rs):
            delivery=[r['diagnostics']['delivery'] for r in rs];outcomes=[r['diagnostics']['outcomes'] for r in rs]
            result['methods'][m]['mechanisms']={'daily_mean_batch_fraction':float(np.mean([r['batch_actions']/r['effective_actions'] for r in delivery if r['effective_actions']])),'pooled_boxes_per_effective_delivery':sum(r['delivered_boxes'] for r in delivery)/sum(r['effective_actions'] for r in delivery),'pooled_expired_after_pick_per_picked':sum(r['expired_after_pick'] for r in outcomes)/sum(r['picked'] for r in outcomes)}
        if all('diagnostics' in r and 'failure_diagnosis' in r['diagnostics'] for r in rs):
            result['failure_patterns'][m]={}
            for l in LOADS:
                result['failure_patterns'][m][l]={}
                for t in TYPES:
                    counts={};urgent={}
                    for r in rs:
                        if r['episode_load']!=l:continue
                        for k,v in r['diagnostics']['failure_diagnosis']['counts_by_type'][t].items():counts[k]=counts.get(k,0)+v
                        for k,v in r['diagnostics']['failure_diagnosis']['urgent_cargo_decision_counts'][t].items():urgent[k]=urgent.get(k,0)+v
                    result['failure_patterns'][m][l][t]={'counts':counts,'urgent_cargo_order_state_counts':urgent}
    pairs=[('urgency','destination'),('coordinated','destination'),('coordinated','urgency')]
    if 'slack' in rows:
        arrays['slack']=data_array(rows['slack'],days);result['methods']['slack']=describe(arrays['slack']);pairs += [(m,'slack') for m in METHODS]
    for left,right in pairs:result['comparisons'][left+' - '+right]=crossed_bootstrap(arrays[left],arrays[right])
    for t in TYPES:
        a={m:data_array(rows[m],days,kind=t) for m in METHODS};result['types'][t]={'methods':{},'comparisons':{}}
        for m,x in a.items():result['types'][t]['methods'][m]={'per_seed_by_load':np.nanmean(x,-1).tolist(),'mean_by_load':np.nanmean(x,axis=(0,2)).tolist(),'eligible_day_ids_by_load':np.isfinite(x[0]).sum(-1).tolist(),'released_order_counts_pooled_by_load':[sum(r['type_counts'][t]['released'] for rs in rows[m] for r in rs if r['episode_load']==l) for l in LOADS]}
        for m in ['urgency','coordinated']:result['types'][t]['comparisons'][m+' - destination']=type_bootstrap(a[m],a['destination'])
    result['paired_daily_differences']=[{'learner_seed':s+1,'load':l,'demand_seed':d,**{m:float(arrays[m][s,i,j]-arrays['destination'][s,i,j]) for m in ['urgency','coordinated']}} for s in range(5) for i,l in enumerate(LOADS) for j,d in enumerate(days)]
    return result


def curves():
    return {m:[{'learner_seed':s,'transitions':[r['transitions'] for r in lines(run_at(m,s)/'evaluations.jsonl')],'on_time':[[r['policies']['ppo']['by_load'][l]['on_time_rate'] for l in LOADS]+[r['policies']['ppo']['mean_on_time_rate']] for r in lines(run_at(m,s)/'evaluations.jsonl')]} for s in range(1,6)] for m in METHODS}


def draw_curves(data,path,tail=False):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7.2),layout='constrained')
    for i,ax in enumerate(axes.flat):
        for m in METHODS:
            v=np.array([r['on_time'] for r in data[m]])[:,:,i]*100;x=np.array(data[m][0]['transitions'])/1000
            for r in v:ax.plot(x,r,color=COLORS[m],alpha=.18,lw=.9)
            ax.plot(x,v.mean(0),color=COLORS[m],linestyle=STYLES[m],lw=2,label=m)
        ax.set(title=[*LOADS,'equal-load mean'][i],xlabel='Training transitions (thousands)',ylabel='Dev daily on-time rate (%)',ylim=(0,102),xlim=(0,1024));ax.grid(alpha=.18)
        if tail:
            v=np.concatenate([np.array([r['on_time'] for r in data[m]])[:,1:,i].ravel() for m in METHODS])*100;ax.set(xlim=(102.4,1024),ylim=(max(0,v.min()-1),min(100.5,v.max()+1)))
        ax.set_xticks([102.4,256,512,768,1024] if tail else [0,256,512,768,1024])
    axes[0,0].legend(frameon=False,loc='lower right');fig.suptitle('Gen3: five learners and mean'+('; from 102,400, focused y scales' if tail else '; 11 fixed Dev checkpoints'))
    for ext in ['png','svg']:fig.savefig(str(path)+'.'+ext,dpi=170)
    plt.close(fig)


def draw_paired(result,path,title):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.9),layout='constrained',sharex=True)
    limits=[]
    for ax,m in zip(axes,['urgency','coordinated']):
        rows=result['comparisons'][m+' - destination']['loads'];point=np.array([rows[l]['estimate'] for l in [*LOADS,'equal']])*100;ci=np.array([rows[l]['joint_95_percentile'] for l in [*LOADS,'equal']])*100;limits.extend(ci.ravel())
        ax.errorbar(point,np.arange(4),xerr=np.stack((point-ci[:,0],ci[:,1]-point)),fmt='o',color=COLORS[m],capsize=4);ax.axvline(0,color='#5d6268',lw=1);ax.set(yticks=np.arange(4),yticklabels=[*LOADS,'equal'],title=m+' - destination',xlabel='On-time rate difference (percentage points)');ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
    span=max(limits)-min(limits);axes[0].set_xlim(min(min(limits),0)-.08*max(span,.1),max(max(limits),0)+.08*max(span,.1))
    fig.suptitle(title+' — paired 95% pointwise intervals')
    for ext in ['png','svg']:fig.savefig(str(path)+'.'+ext,dpi=170)
    plt.close(fig)


def tables(result):
    out=['| Method | Low | Medium | High | Equal |','|---|---:|---:|---:|---:|']
    for m,r in result['methods'].items():out.append('| '+m+' | '+' | '.join(f'{100*v:.2f} ± {100*r["sample_sd"][i]:.2f}' if r['sample_sd'] else f'{100*v:.2f}' for i,v in enumerate(r['mean']))+' |')
    out+=['','Rates in percent; ± is sample SD across five learners, not a confidence interval.','','| Contrast | Equal delta (pp) | Paired joint 95% (pp) | Seed deltas (pp) |','|---|---:|---:|---|']
    for name,d in result['comparisons'].items():
        r=d['loads']['equal'];ci=r['joint_95_percentile'];out.append(f'| {name} | {100*r["estimate"]:+.2f} | [{100*ci[0]:+.2f},{100*ci[1]:+.2f}] | '+', '.join(f'{100*v:+.2f}' for v in r['learner_mean_values'])+' |')
    out+=['','### All type comparisons (exploratory pointwise intervals)','','| Type | Contrast | Low delta [95%] (pp) | Medium | High | Eligible days Low/Med/High |','|---|---|---:|---:|---:|---|']
    for t,d in result['types'].items():
        for name,stats in d['comparisons'].items():
            cells=[]
            for l in LOADS:
                r=stats['loads'][l];ci=r['joint_95_percentile'];cells.append(f'{100*r["estimate"]:+.2f} [{100*ci[0]:+.2f},{100*ci[1]:+.2f}]')
            out.append('| '+t+' | '+name+' | '+' | '.join(cells)+' | '+str(d['methods']['destination']['eligible_day_ids_by_load'])+' |')
    out+=['','Zero-release days omitted consistently; bootstrap day weights renormalized over eligible days per load. Pooled release counts, all subtype means and learner/day components are in JSON. Counts include five policy repetitions of shared demand days, not independent extra days.']
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['dev','test-v3'],required=True);args=parser.parse_args()
    if args.stage=='dev':
        audit=audit_training();write_json(Path('results/gen3_training_audit.json'),audit);rows={m:[] for m in METHODS};last={m:[] for m in METHODS}
        for m in METHODS:
            for s in range(1,6):
                run=run_at(m,s);step=load_checkpoint(run/'best.pt')['steps'];es=lines(run/'evaluation_episodes.jsonl')
                rows[m].append([r for r in es if r['policy']=='ppo' and r['transitions']==step]);last[m].append([r for r in es if r['policy']=='ppo' and r['transitions']==1024000])
        extra={m:[] for m in METHODS};extra_last={m:[] for m in METHODS}
        for m in METHODS:
            for s in range(1,6):
                root='outputs/gen3_baseline_diagnostics' if m=='destination' else f'outputs/gen3_{m}_devextra'
                extra[m].append(read(f'{root}/{m}_s{s}.json')['episodes'])
                oldroot='outputs/gen2_destination_devextra' if m=='destination' else root
                extra_last[m].append(read(f'{oldroot}/{m}_last_s{s}.json')['episodes'])
        result={'dev_best':summarize(rows,list(range(100000,100040))),'dev_last':summarize(last,list(range(100000,100040))),'dev_extra_best':summarize(extra,list(range(100040,100140))),'dev_extra_last':summarize(extra_last,list(range(100040,100140))),'curves':curves()}
        write_json(Path('results/gen3_development.json'),result);draw_curves(result['curves'],Path('results/gen3_learning_curves'));draw_curves(result['curves'],Path('results/gen3_tail_curves'),True);draw_paired(result['dev_extra_best'],Path('results/gen3_dev_extra_paired'),'Gen3 Dev-extra best')
        out=['# Gen3 development: urgency and pickup/delivery coordination','','All prior generations remain frozen. Two separate architecture changes use the same information, K12 actions, reward, PPO, M5 LR schedule and 1,024,000-transition budgets. User confirmed the original overall Dev selection rule; no subtype guardrail is introduced.']
        for stage in ['dev_best','dev_last','dev_extra_best','dev_extra_last']:out+=['','## '+stage,'']+tables(result[stage])
        out+=['','## Interpretation and artifacts','','Dev-extra is development. All three predeclared methods, all five Dev-selected best models each, remain candidates regardless of development ranking. Test-v3 is opened once only after models and statistical code are sealed.','', 'Urgency features first reconstruct physical minutes before cross-type minima; expired physical cargo is counted separately and still retained in the original group embeddings. The coordination variant separately adds the existing 107-dimensional destination group to PICK, including the candidate itself. No urgency+coordination combination or training-objective change is made.','', 'Read-only failure patterns, pooled delivery outcomes, rewards and complete daily paired differences are retained in `gen3_development.json` and raw evaluation outputs. Deadline crossings during an action are observational, not avoidable causal mistakes. All subtype intervals are shown; no multiplicity correction. Five learners and one fixed map limit generalization.']
        Path('results/gen3_development.md').write_text('\n'.join(out)+'\n')
        print(json.dumps({stage:{m:d['mean'] for m,d in result[stage]['methods'].items()} for stage in ['dev_best','dev_extra_best']},indent=2))
    else:
        protocol=read('configs/gen3_test_v3_protocol.json');manifest=read('outputs/gen3_test_v3/manifest.json');assert manifest['status']=='complete' and manifest['protocol_sha256']==sha('configs/gen3_test_v3_protocol.json') and protocol['final_candidates_frozen']
        for p,h in protocol['frozen_files'].items():assert sha(p)==h,p
        rows={m:[] for m in METHODS};rows['slack']=[];total=0;decisions=0
        for entry in protocol['entries']:
            rs=read('outputs/gen3_test_v3/'+entry['name']+'.json')['episodes'];total+=len(rs);decisions+=sum(r['decisions'] for r in rs)
            for r in rs:
                assert r['complete_day'] and r['terminated'] and not r['truncated'] and r['invalid_actions']==0 and r['split']=='test-v3' and 400000<=r['demand_seed']<400100 and r['map_seed']==0
                assert r['released']==r['delivered']+r['expired'] and 0<=r['on_time']<=r['delivered']
                assert all(np.isfinite(r[k]) for k in ['on_time_rate','reward','energy_used','episode_minutes'])
            for policy in entry['policies']:
                selected=[r for r in rs if r['policy']==policy];assert len(selected)==300 and len({(r['episode_load'],r['demand_seed']) for r in selected})==300
            if entry.get('method') in METHODS:
                assert sha(entry['checkpoint'])==entry['checkpoint_sha256'];rows[entry['method']].append(rs)
            else:rows['slack'].append([r for r in rs if r['policy']=='slack'])
        result=summarize(rows,list(range(400000,400100)));result.update(episodes=total,decisions=decisions,invalid_actions=0,wall_seconds=manifest['wall_seconds'],protocol_sha256=sha('configs/gen3_test_v3_protocol.json'),test_v3_status='consumed; no subsequent model/hyperparameter changes or learner-seed selection')
        write_json(Path('results/gen3_test_v3.json'),result);draw_paired(result,Path('results/gen3_test_v3_paired'),'Gen3 new Test-v3 best')
        out=['# Gen3 frozen Test-v3 comparison','','100 new demand IDs400000–400099, fixed before development. All three methods and five selected-best models each were predeclared, irrespective of development ranking. Source/model hashes were sealed before opening Test-v3. The original Test and Test-v2 remain historical only.','']+tables(result)+['','## Boundaries','','Paired crossed learner/day bootstrap:10,000 repetitions, common day weights across loads, pointwise95% intervals, no multiplicity correction. Five learners limit precision. Raw episodes include all outcomes, order counts, resource metrics and diagnostics. Only best checkpoints are tested; stability/last comparisons use Dev. No new-map or optimality claim is supported.','',f'{total} full episodes, {decisions} decisions, no illegal action; suite wall time {manifest["wall_seconds"]:.2f}s. Test-v3 is consumed; further method development needs a new holdout or explicitly labeled development evidence.']
        Path('results/gen3_test_v3.md').write_text('\n'.join(out)+'\n');print(json.dumps({m:d['mean'] for m,d in result['methods'].items()},indent=2))


if __name__=='__main__':main()
