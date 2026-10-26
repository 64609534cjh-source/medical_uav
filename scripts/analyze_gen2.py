"""Audit Gen2 training and report paired development / new holdout comparisons.

Raw episodes/checkpoints remain under outputs. This script never selects a model
using Dev-extra or Test-v2, and never writes first-generation result files.
"""
import argparse,hashlib,json,tarfile
from pathlib import Path
import numpy as np
from medical_uav.training.io import write_json
from medical_uav.evaluation.final_stats import crossed_bootstrap
from medical_uav.training.trainer import load_checkpoint

LOADS=('low','medium','high');METHODS=('m5','destination','relational');TYPES=('plasma','medicine','sample','supplies')
ROOT=Path(__file__).resolve().parents[1] if Path(__file__).parent.name=='scripts' else Path.cwd()


def read(path):return json.loads(Path(path).read_text())
def lines(path):
    with Path(path).open() as f:return [json.loads(x) for x in f]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def m5_runs():
    return {e['learner_seed']:Path(e['checkpoint']).parent for e in read('configs/gen2_holdout_freeze.json')['reference_models']}
def run_at(method,s):return m5_runs()[s] if method=='m5' else Path(f'outputs/gen2_architectures/{method}_s{s}')


def audit_training():
    p=read('configs/gen2_protocol.json');audit=[]
    for path,h in p['frozen_files'].items():assert sha(path)==h,path
    first=read('configs/gen2_holdout_freeze.json')
    for path,h in first['frozen_generation1_files'].items():assert sha(path)==h,path
    for entry in first['reference_models']:assert sha(entry['checkpoint'])==entry['checkpoint_sha256']
    assert read('outputs/gen2_architectures/manifest.json')['status']=='complete'
    for e in p['runs']:
        run=Path('outputs/gen2_architectures')/e['name'];cfg=read(e['config']);meta=read(run/'metadata.json');summary=read(run/'summary.json')
        assert meta['git_dirty'] is False and meta['resume_from'] is None and meta['resume_mode']=='fresh'
        assert meta['config']==cfg==read(run/'config.json')
        source=read(run/'source_manifest.json')
        with tarfile.open(run/'source.tar.gz') as archive:
            for path,h in source.items():assert hashlib.sha256(archive.extractfile(path).read()).hexdigest()==h
        for path,h in p['frozen_files'].items():
            if path.endswith('.py'):assert source[path]==h,(e['name'],path)

        assert summary['status']=='complete' and summary['transitions']==1024000 and summary['updates']==1000
        updates=lines(run/'updates.jsonl');evals=lines(run/'evaluations.jsonl');episodes=lines(run/'evaluation_episodes.jsonl')
        assert len(updates)==1000 and [x['transitions'] for x in evals]==p['evaluation_transitions']
        train_episodes=lines(run/'episodes.jsonl')
        for r in updates:
            assert r['invalid_actions']==0 and r['truncated_count']==0
            for k in ['policy_loss','value_loss','entropy','gradient_norm','initial_ratio_max_error','exact_categorical_kl_mean']:
                assert isinstance(r[k],(float,int)) and np.isfinite(r[k]),(e['name'],r['update_index'],k)
            assert r['initial_ratio_max_error']<1e-5
        assert all(r['invalid_actions']==0 and r['complete_day'] and not r['truncated'] for r in episodes+train_episodes)
        chosen=max(evals,key=lambda r:(r['policies']['ppo']['mean_on_time_rate'],r['policies']['ppo']['mean_reward']))
        best=load_checkpoint(run/'best.pt');last=load_checkpoint(run/'last.pt')
        assert best['steps']==chosen['transitions'] and last['steps']==1024000
        for r in evals:
            cp=load_checkpoint(run/r['checkpoint']);assert cp['steps']==r['transitions']
            assert all(np.isfinite(v.numpy()).all() for v in cp['model_state_dict'].values())
            raw=[x for x in episodes if x['policy']=='ppo' and x['transitions']==r['transitions']]
            assert len(raw)==120
            score=np.mean([np.mean([x['on_time_rate'] for x in raw if x['episode_load']==l]) for l in LOADS])
            assert abs(score-r['policies']['ppo']['mean_on_time_rate'])<1e-12
        audit.append({'name':e['name'],'source_commit':meta['git_commit'],'best_steps':best['steps'],'evaluations':len(evals),'evaluation_days':len(episodes),'updates':len(updates),'train_complete_days':len(train_episodes),'invalid_actions':0,'finite_training_and_all_checkpoint_parameters':True,'wall_seconds':summary['wall_seconds'],'training_seconds':summary['training_seconds'],'eval_seconds':summary['eval_seconds'],'transitions_by_load':summary['transitions_by_load'],'episodes_by_load':summary['episodes_by_load'],'source_manifest_sha256':sha(run/'source_manifest.json'),'best_sha256':sha(run/'best.pt'),'last_sha256':sha(run/'last.pt'),
            'tail_200_updates':{k:float(np.mean([r[k] for r in updates[-200:]])) for k in ['value_loss','entropy','exact_categorical_kl_mean','old_normalized_entropy_choice_mean','gradient_norm','kl_early_stop','optimizer_steps']}})
    return {'runs':audit,'frozen_gen1_files_verified':len(first['frozen_generation1_files']),'formal_source_commits':sorted({r['source_commit'] for r in audit}),'budget_total':10240000,'all_checks_passed':True}


def data_array(rows_by_seed,days,field='on_time_rate',kind=None):
    result=[]
    for rows in rows_by_seed:
        lookup={(r['episode_load'],r['demand_seed']):r for r in rows}
        assert len(lookup)==len(rows)==3*len(days)
        matrix=[]
        for load in LOADS:
            row=[]
            for d in days:
                r=lookup[load,d];assert r['complete_day'] and not r['invalid_actions'] and not r['truncated']
                if kind:
                    counts=r['type_counts'][kind];row.append(counts['on_time']/counts['released'] if counts['released'] else np.nan)
                else:row.append(r[field])
            matrix.append(row)
        result.append(matrix)
    return np.asarray(result,dtype=float)


def describe(a):
    v=np.concatenate((a.mean(-1),a.mean((1,2))[:,None]),axis=1)
    return {'per_seed':v.tolist(),'mean':v.mean(0).tolist(),'sample_sd':v.std(0,ddof=1).tolist() if len(a)>1 else None,'columns':[*LOADS,'equal']}


def summarize_set(rows,days):
    arrays={m:data_array(rows[m],days) for m in METHODS}
    baseline={(r['episode_load'],r['demand_seed']):(r['released'],{k:c['released'] for k,c in r['type_counts'].items()}) for r in rows['m5'][0]}
    for method in METHODS:
        for rs in rows[method]:
            for r in rs:assert baseline[r['episode_load'],r['demand_seed']]==(r['released'],{k:c['released'] for k,c in r['type_counts'].items()})
    results={'methods':{m:describe(a) for m,a in arrays.items()},'comparisons':{},'per_type':{}}
    pairs=[('destination','m5'),('relational','m5'),('relational','destination')]
    if 'slack' in rows:
        arrays['slack']=data_array(rows['slack'],days);results['methods']['slack']=describe(arrays['slack'])
        pairs += [(m,'slack') for m in METHODS]
    for left,right in pairs:results['comparisons'][left+' - '+right]=crossed_bootstrap(arrays[left],arrays[right])
    for kind in TYPES:
        type_arrays={m:data_array(rows[m],days,kind=kind) for m in METHODS}
        valid=np.isfinite(type_arrays['m5'][0])
        for a in type_arrays.values():assert np.array_equal(np.isfinite(a),np.broadcast_to(valid,a.shape))
        # The daily type denominator depends only on demand, not the policy.
        results['per_type'][kind]={'eligible_days_by_load':valid.sum(-1).tolist(),'methods':{},'paired_delta_pp':{}}
        for m,a in type_arrays.items():
            per=np.nanmean(a,axis=-1)
            results['per_type'][kind]['methods'][m]={'per_seed_by_load':per.tolist(),'mean_by_load':per.mean(0).tolist(),'sample_sd_by_load':per.std(0,ddof=1).tolist()}
        for m in ['destination','relational']:
            results['per_type'][kind]['paired_delta_pp'][m]=(100*np.nanmean(type_arrays[m]-type_arrays['m5'],axis=(0,2))).tolist()
    # Complete matched daily differences, not only group means.
    results['paired_daily_differences']=[{'learner_seed':s+1,'load':l,'demand_seed':day,**{m:float(arrays[m][s,i,j]-arrays['m5'][s,i,j]) for m in ['destination','relational']}} for s in range(5) for i,l in enumerate(LOADS) for j,day in enumerate(days)]
    results['mechanisms']={}
    for m in METHODS:
        rs=[r for group in rows[m] for r in group]
        if all('diagnostics' in r for r in rs):
            delivery=[r['diagnostics']['delivery'] for r in rs]
            outcomes=[r['diagnostics']['outcomes'] for r in rs]
            fractions=[d['batch_actions']/d['effective_actions'] for d in delivery if d['effective_actions']]
            effective=sum(d['effective_actions'] for d in delivery)
            picked=sum(o['picked'] for o in outcomes)
            results['mechanisms'][m]={'episodes':len(rs),'daily_mean_batch_fraction':float(np.mean(fractions)),
                'days_with_effective_delivery':len(fractions),
                'boxes_per_effective_delivery_pooled':sum(d['delivered_boxes'] for d in delivery)/effective if effective else None,
                'expired_after_pick_fraction_pooled':sum(o['expired_after_pick'] for o in outcomes)/picked if picked else None,
                'daily_mean_wait_minutes':float(np.mean([r['wait_minutes'] for r in rs])),
                'daily_mean_flight_energy':float(np.mean([r['energy_used'] for r in rs]))}
        results['methods'][m]['reward']=describe(data_array(rows[m],days,'reward'))
        results['methods'][m]['served_rate']=describe(data_array(rows[m],days,'served_rate'))
        results['methods'][m]['expiry_rate']=describe(data_array(rows[m],days,'expiry_rate'))
    return results


def curves():
    result={}
    for m in METHODS:
        result[m]=[]
        for s in range(1,6):
            es=lines(run_at(m,s)/'evaluations.jsonl')
            result[m].append({'learner_seed':s,'transitions':[e['transitions'] for e in es],
                'on_time':[[e['policies']['ppo']['by_load'][l]['on_time_rate'] for l in LOADS]+[e['policies']['ppo']['mean_on_time_rate']] for e in es],
                'reward':[e['policies']['ppo']['mean_reward'] for e in es]})
    return result


def draw_curves(data,path,tail=False):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'m5':'#657084','destination':'#2574a9','relational':'#ba7423'}
    fig,axes=plt.subplots(2,2,figsize=(11,7.2),layout='constrained')
    for j,ax in enumerate(axes.flat):
        for m in METHODS:
            values=np.array([r['on_time'] for r in data[m]])[:,:,j]*100
            x=np.array(data[m][0]['transitions'])/1000
            for row in values:ax.plot(x,row,color=colors[m],alpha=.18,lw=.9)
            ax.plot(x,values.mean(0),color=colors[m],lw=2,label=m)
        ax.set(title=[*LOADS,'equal-load mean'][j],xlabel='Training transitions (thousands)',ylabel='Dev-main daily on-time rate (%)',ylim=(0,102),xlim=(0,1024));ax.grid(alpha=.18)
        if tail:
            focus=np.concatenate([np.array([r['on_time'] for r in data[m]])[:,1:,j].ravel() for m in METHODS])*100
            ax.set_xlim(102.4,1024);ax.set_ylim(max(0,focus.min()-1),min(100.5,focus.max()+1))
    axes[0,0].legend(frameon=False,loc='lower right');fig.suptitle('Gen2: all five learners and mean; '+('from 102,400 transitions, focused y scales' if tail else '11 fixed Dev checkpoints'))
    for ext in ['png','svg']:fig.savefig(str(path)+'.'+ext,dpi=170)
    plt.close(fig)


def draw_paired(result,path,title):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.9),layout='constrained')
    for ax,name,color in zip(axes,['destination - m5','relational - m5'],['#2574a9','#ba7423']):
        rs=result['comparisons'][name]['loads'];y=np.arange(4)
        point=np.array([rs[l]['estimate'] for l in [*LOADS,'equal']])*100
        ci=np.array([rs[l]['joint_95_percentile'] for l in [*LOADS,'equal']])*100
        ax.errorbar(point,y,xerr=np.stack((point-ci[:,0],ci[:,1]-point)),fmt='o',color=color,capsize=4)
        ax.axvline(0,color='#5d6268',lw=1);ax.set(yticks=y,yticklabels=[*LOADS,'equal'],title=name,xlabel='On-time rate difference (percentage points)');ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
    fig.suptitle(title+' — paired 95% pointwise crossed-bootstrap intervals')
    for ext in ['png','svg']:fig.savefig(str(path)+'.'+ext,dpi=170)
    plt.close(fig)


def congestion(rows):
    result={}
    for m in METHODS:
        result[m]={}
        for load in LOADS:
            per=[];total={}
            for rs in rows[m]:
                selected=[r for r in rs if r['episode_load']==load];fractions=[];scores=[]
                for r in selected:
                    c=r['diagnostics']['congestion']
                    for k,v in c.items():total[k]=max(total.get(k,0),v) if k=='max_pending' else total.get(k,0)+v
                    fractions.append(c['over_12_states']/c['states']);scores.append(r['on_time_rate'])
                corr=float(np.corrcoef(fractions,scores)[0,1]) if np.std(fractions)>0 and np.std(scores)>0 else None
                per.append({'correlation_over12_fraction_vs_daily_ontime':corr,'day_mean_over12_fraction':float(np.mean(fractions))})
            result[m][load]={'counts':total,'over12_state_fraction':total['over_12_states']/total['states'],'hidden_partner_state_fraction':total['hidden_matching_visible_or_cargo_states']/total['states'],'hidden_direct_on_time_partner_state_fraction':total['hidden_direct_on_time_matching_states']/total['states'],'per_learner':per}
    return {'scope':'Dev-extra; decision-state weighting across learners; correlations calculated separately per learner/load; full-simulator read-only instrumentation; no causal conclusion or K24 test','methods':result}


def markdown_table(result):
    rows=['| Method | Low | Medium | High | Equal |','|---|---:|---:|---:|---:|']
    for m,d in result['methods'].items():
        vals=[f'{100*v:.2f} ± {100*d["sample_sd"][i]:.2f}' if d['sample_sd'] is not None else f'{100*v:.2f}' for i,v in enumerate(d['mean'])]
        rows.append('| '+m+' | '+' | '.join(vals)+' |')
    rows+=['','Rates in percent; ± is sample SD across five independent learners, not a confidence interval.','', '| Paired contrast | Equal delta (pp) | Joint 95% interval (pp) | Seed deltas (pp) |','|---|---:|---:|---|']
    for name,d in result['comparisons'].items():
        r=d['loads']['equal'];ci=r['joint_95_percentile'];rows.append(f'| {name} | {100*r["estimate"]:+.2f} | [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}] | '+', '.join(f'{100*x:+.2f}' for x in r['learner_mean_values'])+' |')
    return rows


def report_types(result):
    out=['','### Daily order-type differences from M5','', '| Type | Method | Low (pp) | Medium (pp) | High (pp) | Eligible days Low/Med/High |','|---|---|---:|---:|---:|---|']
    for kind,d in result['per_type'].items():
        for m,delta in d['paired_delta_pp'].items():out.append('| '+kind+' | '+m+' | '+' | '.join(f'{v:+.2f}' for v in delta)+' | '+str(d['eligible_days_by_load'])+' |')
    out+=['','Daily type rates omit only demand days with no releases of that type. Matched availability is verified across methods. These subtype comparisons are descriptive; no simultaneous-inference claim.']
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['dev','test-v2'],required=True);args=p.parse_args()
    if args.stage=='dev':
        audit=audit_training();write_json(Path('results/gen2_training_audit.json'),audit)
        rows={m:[] for m in METHODS};last={m:[] for m in METHODS}
        for m in METHODS:
            for s in range(1,6):
                run=run_at(m,s);step=load_checkpoint(run/'best.pt')['steps'];es=lines(run/'evaluation_episodes.jsonl')
                rows[m].append([r for r in es if r['policy']=='ppo' and r['transitions']==step])
                last[m].append([r for r in es if r['policy']=='ppo' and r['transitions']==1024000])
        result={'dev_best':summarize_set(rows,list(range(100000,100040))),'dev_last':summarize_set(last,list(range(100000,100040))),'curves':curves()}
        extra={m:[] for m in METHODS};extra_last={m:[] for m in METHODS}
        for m in METHODS:
            for s in range(1,6):
                root='outputs/gen2_m5_devextra' if m=='m5' else f'outputs/gen2_{m}_devextra'
                extra[m].append(read(f'{root}/{m}_s{s}.json')['episodes'])
                if m!='m5':extra_last[m].append(read(f'{root}/{m}_last_s{s}.json')['episodes'])
        result['dev_extra_best']=summarize_set(extra,list(range(100040,100140)))
        # Baseline last scores already measured in M5, reused without selecting anew.
        for s in range(1,6):
            extra_last['m5'].append(read(f'outputs/m5_s1_devextra_all/s1_last_s{s}.json')['episodes'])
        result['dev_extra_last']=summarize_set(extra_last,list(range(100040,100140)))
        result['congestion']=congestion(extra);write_json(Path('results/gen2_development.json'),result)
        draw_curves(result['curves'],Path('results/gen2_learning_curves'))
        draw_curves(result['curves'],Path('results/gen2_tail_curves'),tail=True)
        draw_paired(result['dev_extra_best'],Path('results/gen2_dev_extra_paired'),'Gen2 Dev-extra, selected best')
        out=['# Gen2: independent destination and relational architecture experiments','','Generation 1 remains frozen at 3c05ad3. Training sources/configurations were sealed before ten scratch runs. Neither new architecture changes observations, actions, reward, PPO, candidate K=12, or the M5 learning-rate schedule. All ten runs complete 1,024,000 transitions, with 11 fixed Dev checkpoints.','', '## Dev-main selected best','']+markdown_table(result['dev_best'])+['','## Dev-main last','']+markdown_table(result['dev_last'])+['','## Dev-extra selected best','']+markdown_table(result['dev_extra_best'])+['','## Dev-extra last','']+markdown_table(result['dev_extra_last'])+report_types(result['dev_extra_best'])
        out+=['','## Top-12 diagnostic (Dev-extra, selected best)','','| Method | Load | States >12 pending | Hidden same-destination partner | Direct-on-time hidden partner | Maximum pending |','|---|---|---:|---:|---:|---:|']
        for m in METHODS:
            for l,r in result['congestion']['methods'][m].items():out.append(f'| {m} | {l} | {100*r["over12_state_fraction"]:.2f}% | {100*r["hidden_partner_state_fraction"]:.2f}% | {100*r["hidden_direct_on_time_partner_state_fraction"]:.2f}% | {r["counts"]["max_pending"]} |')
        out+=['','Hidden partner means destination matches visible pending or physical cargo. Direct-on-time is a one-order time-window check, without a full capacity/energy route feasibility test. Correlations and state frequencies indicate exposure, not causal performance loss. They do not establish that K=24 will improve service. No K=24 or combined network is trained.','', '## Reproducibility','','- Training audit: `gen2_training_audit.json`; complete configurations and source hashes: `../configs/gen2_protocol.json`.','- Best is selected only on 40 Dev-main days × three loads. Dev-extra is development, not independent Test. Last is retained without selection.','- Per-day differences, reward, delivered/expired metrics, learner/day-only intervals, type denominators and all curves are in `gen2_development.json`.','- Five learner IDs and demand-day IDs are paired across methods. Intervals use 10,000 product-bootstrap repetitions; day weights are shared across loads. Intervals are pointwise, without familywise correction. Only five learners and one fixed map limit generalization.','- All three methods, regardless of development ranking, remain predeclared Test-v2 candidates. Test-v2 is opened only after hashes of all 15 models and final statistical code are frozen.','', '## Architecture references','','The designs test specific inductive biases; they are not exact reproductions of these papers and the papers do not guarantee benefits in this simulator. [Deep Sets](https://arxiv.org/abs/1703.06114), [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html).']
        if result['dev_extra_best']['mechanisms']:
            out+=['','## Delivery mechanism diagnostics (Dev-extra best)','','| Method | Mean daily batch DELIVER fraction | Pooled boxes / effective DELIVER | Pooled expired-after-pick / picked |','|---|---:|---:|---:|']
            for m,r in result['dev_extra_best']['mechanisms'].items():out.append(f'| {m} | {100*r["daily_mean_batch_fraction"]:.2f}% | {r["boxes_per_effective_delivery_pooled"]:.3f} | {100*r["expired_after_pick_fraction_pooled"]:.2f}% |')
            out+=['','Descriptive associations under each learned policy, not causal attribution to a network component. Effective DELIVER excludes actions delivering no box.']
        Path('results/gen2_development.md').write_text('\n'.join(out)+'\n')
        print(json.dumps({'best':result['dev_best']['methods'],'extra':result['dev_extra_best']['methods']},indent=2))
    else:
        protocol=read('configs/gen2_test_v2_protocol.json');manifest=read('outputs/gen2_test_v2/manifest.json');assert manifest['status']=='complete' and manifest['protocol_sha256']==sha('configs/gen2_test_v2_protocol.json')
        assert protocol['final_candidates_frozen'] and protocol['split']=='test-v2'
        for p,h in protocol['frozen_files'].items():assert sha(p)==h,p
        rows={m:[] for m in METHODS};rows['slack']=[];total=0;decisions=0
        for e in protocol['entries']:
            result=read('outputs/gen2_test_v2/'+e['name']+'.json');rs=result['episodes'];total+=len(rs);decisions+=sum(r['decisions'] for r in rs)
            for r in rs:
                assert r['complete_day'] and r['terminated'] and not r['truncated'] and r['invalid_actions']==0
                assert r['split']=='test-v2' and 300000<=r['demand_seed']<300100 and r['map_seed']==0
                assert r['released']==r['delivered']+r['expired'] and 0<=r['on_time']<=r['delivered']
                assert all(np.isfinite(r[k]) for k in ['on_time_rate','reward','energy_used','episode_minutes'])
            for policy in e['policies']:
                selected=[r for r in rs if r['policy']==policy]
                assert len(selected)==300 and len({(r['episode_load'],r['demand_seed']) for r in selected})==300
            if e.get('method') in METHODS:
                assert sha(e['checkpoint'])==e['checkpoint_sha256'];rows[e['method']].append(rs)
            else:rows['slack'].append([r for r in rs if r['policy']=='slack'])
        result=summarize_set(rows,list(range(300000,300100)));result.update(episodes=total,decisions=decisions,invalid_actions=0,wall_seconds=manifest['wall_seconds'],protocol_sha256=sha('configs/gen2_test_v2_protocol.json'),test_v2_status='consumed; no post-Test-v2 tuning performed')
        write_json(Path('results/gen2_test_v2.json'),result);draw_paired(result,Path('results/gen2_test_v2_paired'),'Gen2 new Test-v2, selected best')
        out=['# Gen2: preregistered new Test-v2 comparison','','Test-v2 contains 100 previously unused demand days (300000–300099), at Low/Medium/High, fixed map 0. Day IDs were sealed before architecture development. All 15 selected-best models and statistical source hashes were sealed before Test-v2 was opened. The original Test is retained as historical Generation 1 evidence.','']+markdown_table(result)+report_types(result)
        out+=['','## Interpretation boundaries','','The primary contrasts are destination − M5 and relational − M5. The architecture comparisons are pointwise 95% intervals, without familywise correction; treat fine-grained multiple comparisons as exploratory. Paired learner-ID/day-ID product bootstrap uses 10,000 repetitions, common demand-day resampling across loads and separate learner/day-only intervals in the JSON. Five learners do not provide a precise estimate of all training variability.','', 'Only selected-best checkpoints are evaluated here. Training stability and last-checkpoint comparisons use Dev-main/Dev-extra, not Test-v2. No inference about a new map or theoretical optimum is supported. Absence of improvement is a valid result; neither architecture was combined with the other or tuned after this comparison.','', f'{total} full days, {decisions} decisions, no illegal action; suite wall time {manifest["wall_seconds"]:.2f} seconds. Complete raw per-day metrics and type outcomes remain in `../outputs/gen2_test_v2/`.','', 'Test-v2 is now consumed. Further method selection requires a new holdout or a clearly labeled development comparison.']
        Path('results/gen2_test_v2.md').write_text('\n'.join(out)+'\n');print(json.dumps(result['methods'],indent=2))


if __name__=='__main__':main()
