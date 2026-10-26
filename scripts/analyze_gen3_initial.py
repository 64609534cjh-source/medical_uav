"""All-type Dev-only failure analysis before Gen3 scratch training."""
import json
from pathlib import Path
import numpy as np
from medical_uav.evaluation.type_stats import type_bootstrap
from medical_uav.training.io import write_json

LOADS=('low','medium','high');TYPES=('plasma','medicine','sample','supplies')


def load_rows(method):
    return [json.loads(Path(f'outputs/gen3_baseline_diagnostics/{method}_s{s}.json').read_text())['episodes'] for s in range(1,6)]


def outcome_array(rows,kind,outcome):
    values=[]
    for rs in rows:
        by={(r['episode_load'],r['demand_seed']):r for r in rs};assert len(by)==300
        matrix=[]
        for l in LOADS:
            entries=[]
            for d in range(100040,100140):
                o=by[l,d]['diagnostics']['outcomes_by_type'][kind]
                entries.append(o[outcome]/o['released'] if o['released'] else np.nan)
            matrix.append(entries)
        values.append(matrix)
    return np.asarray(values)


def main():
    manifest=json.loads(Path('outputs/gen3_baseline_diagnostics/manifest.json').read_text());assert manifest['status']=='complete'
    rows={m:load_rows(m) for m in ['m5','destination']};verified=0
    for m in rows:
        oldroot=f'outputs/gen2_{m}_devextra'
        for s,rs in enumerate(rows[m],1):
            old={(r['episode_load'],r['demand_seed']):r for r in json.loads(Path(f'{oldroot}/{m}_s{s}.json').read_text())['episodes']}
            for r in rs:
                clean=dict(r);clean['diagnostics']=dict(r['diagnostics']);clean['diagnostics'].pop('failure_diagnosis')
                assert clean==old[r['episode_load'],r['demand_seed']];verified+=1
    output={'stage':'Gen3 pre-training Dev-extra diagnostic','unchanged_full_day_records_verified':verified,'selection':'user confirmed original overall Dev rule; no service/reward guardrails','types':{},'failure_counts':{},'scope':'exploratory pointwise intervals, no multiplicity correction; five independent learners; 100 shared demand-day IDs; diagnostic full-simulator fields never supplied to policies'}
    out=['# Gen3 pre-training: all-type failure diagnosis (Dev-extra only)','','Frozen M5 and Gen2 destination policies were replayed on the same 100 Dev-extra days × three loads × five learners. After removing the new failure instrumentation, every original day field and diagnostic is exactly unchanged. No original Test or Test-v2 was rerun.','','| Type | Load | Eligible days | Destination − M5 on-time (pp, paired 95%) | Late delta | Expired unpicked delta | Expired after pick delta |','|---|---|---:|---|---:|---:|---:|']
    for t in TYPES:
        arrays={m:{o:outcome_array(rows[m],t,o) for o in ['on_time','late','expired_unpicked','expired_after_pick']} for m in rows}
        stats={o:type_bootstrap(arrays['destination'][o],arrays['m5'][o]) for o in arrays['m5']};output['types'][t]=stats
        for i,l in enumerate(LOADS):
            r=stats['on_time']['loads'][l];delta=[stats[o]['loads'][l]['estimate'] for o in ['late','expired_unpicked','expired_after_pick']]
            assert abs(r['estimate']+sum(delta))<1e-12
            ci=r['joint_95_percentile'];out.append(f'| {t} | {l} | {r["eligible_days"]} | {100*r["estimate"]:+.2f} [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}] | '+' | '.join(f'{100*v:+.2f}' for v in delta)+' |')
    for m in rows:
        output['failure_counts'][m]={}
        for l in LOADS:
            output['failure_counts'][m][l]={}
            for t in TYPES:
                selected=[r for group in rows[m] for r in group if r['episode_load']==l];counts={};opportunities={}
                for r in selected:
                    for k,v in r['diagnostics']['failure_diagnosis']['counts_by_type'][t].items():counts[k]=counts.get(k,0)+v
                    for k,v in r['diagnostics']['failure_diagnosis']['urgent_cargo_decision_counts'][t].items():opportunities[k]=opportunities.get(k,0)+v
                output['failure_counts'][m][l][t]={'counts':counts,'urgent_cargo_order_state_counts':opportunities,'unique_day_ids':100,'learner_repetitions':5}
    out+=['','All outcome deltas are mean daily type rates; zero-release days are omitted consistently. On-time + late + expired-unpicked + expired-after-pick differences sum to zero exactly. Interval resampling pairs learner/day IDs, sharing day weights across loads and renormalizing weights over eligible days. Counts, separate learner/day uncertainty, all outcomes and per-order failure details are retained in JSON/raw outputs.','','Deadline crossings during PICK/DELIVER/CHARGE/WAIT/RETURN describe the action underway, not an avoidable causal error. Direct slack ignores alternative routes, capacity and energy. First-visible status and never-visible failures are diagnostics only; hidden simulator order details do not enter model inputs. Counts aggregate five repetitions of the same demand days, not independent demand samples.','','The original destination summary contains nonlinear mean cargo/order embeddings and counts, not explicit physical-time minima. This motivates the fixed urgency statistics without claiming that learned embeddings lose all urgency information. The coordination experiment separately tests whether PICK benefits from the existing destination context. No reward weights, per-type tolerance or model-selection guardrail is introduced.']
    write_json(Path('results/gen3_initial_diagnostics.json'),output);Path('results/gen3_initial_diagnostics.md').write_text('\n'.join(out)+'\n')
    print('Verified unchanged records:',verified)
    for t in TYPES:
        print(t,{l:{'delta_pp':100*r['estimate'],'CI_pp':[100*v for v in r['joint_95_percentile']]} for l,r in output['types'][t]['on_time']['loads'].items()})


if __name__=='__main__':main()
