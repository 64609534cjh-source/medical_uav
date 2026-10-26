"""Summarize the finite tuning decision, optional inference optimization, and Test."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1] if Path(__file__).parent.name=='scripts' else Path.cwd()
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from medical_uav.training.io import write_json


def read(p):return json.loads(Path(p).read_text())


def main():
    gate=read(ROOT/'results/m5_development_gate.json')
    tuning=read(ROOT/'results/m5_tuning.json') if (ROOT/'results/m5_tuning.json').exists() else gate
    final=read(ROOT/'results/final_test.json');bench=read(ROOT/'results/m5_inference.json')
    selected=final['protocol']['final_method'];s0=read(ROOT/'results/m4_structured_audit.json');seeds=tuning['seeds']
    result={'milestone':'M5','development_gate':gate,'tuning':tuning,'inference':bench,'final_method':selected,
            'test_result':'final_test.json','test_protocol_sha256':final['protocol_sha256'],
            'm4_preserved':True,'experiments_run':['S1 late LR only'],'experiments_not_run':['S2 stricter KL (not supported by diagnosis)','combined/entropy/reward/network changes'],
            'optimization_separate_from_training':True,'test_after_freeze':True}
    write_json(ROOT/'results/m5_summary.json',result)
    lines=['# M5: bounded stability tuning, inference reuse and frozen Test','',
           'M4 S0 was preserved. Only the preregistered S1 late-learning-rate schedule was trained; no architecture/reward/entropy/slot change or combination search was added. S2 was not run because the pre-tuning diagnosis did not motivate stricter update control.','',
           '## Independent development-day diagnosis','',
           'Dev-extra 100040–100139 was unused in recorded local experiments before M5. It is now explicitly part of development, not Test. Existing M4 best/last, MLP best/last and Slack were evaluated before tuning. M4 best is 85.96%, MLP best 77.45%, Slack 81.47%; M4 last is 84.51%. All five M4 best-to-last declines repeat on the extra days. These results support investigating a policy-level tradeoff, while the original best/main gap also contains selection optimism.','',
           'The [initial diagnostic](m5_initial_diagnostics.md) retains per-seed/load/type rates and counts, unpicked/picked expiry, effective-drop denominators, transport/resource measurements, LR and KL stopping. M4 main results were exactly reproduced.','',
           '## Single-factor schedule test','',
           'S1 keeps the actual M4 constant LR 3e-4 through update 500, then decays linearly to 3e-5 at update 1000. Every run starts from scratch and completes 1,024,000 transitions with the same 11 Dev-main checkpoints and same selection rule. Midpoint model tensors and collector sampler/counters exactly reproduce M4; the golden updater check shows that added diagnostics leave original updates unchanged. Exact categorical KL and legal-count-normalized entropy are now logged, without changing joint KL stopping.','',
           '| Development gate (seeds 1–3) | Observed change (pp) | Required change (pp) | Pass |','|---|---:|---:|---|']
    proto=read(ROOT/'configs/m5_protocol.json')['extension_gate']
    mappings=[('best_main','main_deltas','best','best_main_delta_at_least'),('last_main','main_deltas','last','last_main_delta_at_least'),('tail_main','main_deltas','tail','tail_main_delta_at_least'),('best_extra','extra_deltas','best','best_extra_delta_at_least'),('last_extra','extra_deltas','last','last_extra_delta_at_least')]
    for name,group,key,limit in mappings:lines.append(f"| {name} | {100*gate[group][key]['mean_delta']:+.2f} | ≥ {100*proto[limit]:+.2f} | {gate['gate_checks'][name]} |")
    lines += [f"| minimum per-load extra best | {100*min(gate['extra_best_load_deltas'].values()):+.2f} | ≥ −2.00 | {gate['gate_checks']['per_load_extra']} |",f"| minimum per-type extra best | {100*min(v for d in gate['extra_best_type_deltas'].values() for v in d.values()):+.2f} | ≥ −3.00 | {gate['gate_checks']['per_type_extra']} |",'',
              'The complete preregistered gate '+('passed, so scratch seeds 4–5 were also completed.' if gate['gate_pass'] else 'did not pass, so tuning stopped and M4 was retained; no seeds 4–5 were added.'),
              'These are fixed practical selection tolerances, not significance tests or statistical noninferiority claims. The final method was selected before Test.','',
              '| S1 seed | Best step | Best % | Last % | Tail % | Main best change (pp) | Main last change (pp) |','|---:|---:|---:|---:|---:|---:|---:|']
    for i,a in enumerate(tuning['audits']):lines.append(f"| {a['seed']} | {a['best']['steps']:,} | {100*a['best']['mean_on_time_rate']:.2f} | {100*a['last']['mean_on_time_rate']:.2f} | {100*a['tail']:.2f} | {100*tuning['main_deltas']['best']['deltas'][i]:+.2f} | {100*tuning['main_deltas']['last']['deltas'][i]:+.2f} |")
    lines += ['', '| Extra best service tradeoff (pp) | Plasma | Medicine | Sample | Supplies |', '|---|---:|---:|---:|---:|']
    for load in ('low','medium','high'):
        d=tuning['extra_best_type_deltas'][load]
        lines.append('| '+load+' | '+' | '.join(f"{100*d[k]:+.2f}" for k in ('plasma','medicine','sample','supplies'))+' |')
    lines += ['', 'These are on-time daily-rate changes on the explicitly developmental extra set, using the same selected best checkpoint per learner. Overall stability improvement does not imply that every order type improves; the supplies tradeoff is retained rather than hidden by the equal-load mean.']
    lines += ['', 'Each tail is the arithmetic mean of existing checkpoints 819200, 921600 and 1024000; it creates no extra selection opportunities. Per-load/type extra comparisons, actual LR, exact/approximate KL, normalized entropy, optimizer steps, early-stop counts and complete source/checkpoint audits remain in the JSON.','',
              '## Inference-only engineering change','',
              'The training network is unchanged. A separate inference subclass reuses the original PICK layer weights as Wc·context + We·order + bias; context projection is computed once. An action-only path bypasses categorical log-probability/entropy work. Near legal-logit ties (gap ≤1e-5), it falls back to the original categorical decision. Value outputs and state_dict keys remain compatible; float PICK logits are checked numerically, not claimed bit-identical.','',
              '| Path | B=1 p50 / p95 (µs) | B=8 p50 / p95 (µs per batch) | B=8 states/s at p50 |','|---|---:|---:|---:|']
    for name,r in bench['latency'].items():
        a,b=r['1'],r['8'];lines.append(f"| {name} | {a['p50_us_per_batch']:.2f} / {a['p95_us_per_batch']:.2f} | {b['p50_us_per_batch']:.2f} / {b['p95_us_per_batch']:.2f} | {b['states_per_second_at_p50']:.0f} |")
    lines += ['',f"Full Dev (120 days, {bench['full_dev']['original']['decisions']:,} decisions): original median {bench['full_dev']['original']['median_seconds']:.3f} s, factored argmax median {bench['full_dev']['factored_argmax']['median_seconds']:.3f} s across three repetitions. One CPU thread, identical archived real observations, 200 warmups and 2,000 individually timed calls per microbenchmark. No learner or other evaluation suite was active.",
              f"Equivalence was checked over {bench['validated_days']:,} complete Dev days across {len(bench['validated_models'])} preselected checkpoints: every action and every complete-day record field reproduced the canonical results. Maximum measured logit error: {max(r['logit_max_abs_error'] for r in bench['validated_models']):.3g}; value error: {max(r['value_max_abs_error'] for r in bench['validated_models']):.3g}.",
              'Linear MAC estimates are 139,392 (MLP), 267,584 (original structured), 177,472 (factored structured); 90,112 repeated MACs are removed. These are analytic operation counts, not measured latency or memory. On B=1, argmax-only is slightly faster than the factored variant; extra small-tensor calls offset the saved MACs. B=8 is similar. The main latency reduction comes from the complete action-only inference path, and no proportional MAC-to-latency speedup is claimed. The scientific Test suite uses the canonical evaluator, so engineering and learning changes remain distinct.','',
              '## Frozen final Test','', 'Final selected structural method: **'+selected.upper()+'**. All models, checkpoint hashes, comparisons, metrics and the paired learner/day bootstrap were committed before any Test evaluation.','', '| Method | Test equal on-time % | Learner SD (pp) | Joint 95% interval % |','|---|---:|---:|---:|']
    for name,r in final['performance'].items():
        v=r['loads']['equal'];ci=v['joint_95_percentile'];sd='—' if v['learner_sample_sd'] is None else f"{100*v['learner_sample_sd']:.2f}"
        lines.append(f"| {name} | {100*v['estimate']:.2f} | {sd} | [{100*ci[0]:.2f}, {100*ci[1]:.2f}] |")
    final_vs_slack=final['comparisons'][selected+'_minus_slack']['loads']['equal']
    architecture=final['comparisons']['m4_minus_mlp']['loads']['equal']
    lo,hi=final_vs_slack['joint_95_percentile']
    lines += ['', f"The original same-PPO architecture advantage repeats on Test: M4 − MLP is {100*architecture['estimate']:+.2f} pp. Final {selected.upper()} − Slack is {100*final_vs_slack['estimate']:+.2f} pp, joint 95% interval [{100*lo:+.2f}, {100*hi:+.2f}] pp."]
    if selected=='s1':
        change=final['comparisons']['s1_minus_m4']['loads']['equal'];a,b=change['joint_95_percentile']
        lines += [f"S1 − M4 on Test is {100*change['estimate']:+.2f} pp, with interval [{100*a:+.2f}, {100*b:+.2f}] pp. It includes zero: there is no clear additional selected-checkpoint performance gain. The demonstrated M5 benefit is the predeclared main/extra development-set last/tail stability improvement, while retaining the architecture's Test advantage."]
    low=final['comparisons'][selected+'_minus_slack']['loads']['low']
    lines += [f"Low does not show a clear advantage over Slack ({100*low['estimate']:+.2f} pp; interval [{100*low['joint_95_percentile'][0]:+.2f}, {100*low['joint_95_percentile'][1]:+.2f}] pp). Medium and High provide the positive final aggregate gain. No claim is made that every load or order type improves."]
    lines += ['', 'The [final Test report](final_test.md) includes all learner/load/type results and paired comparison intervals, with seed-only, day-only and joint uncertainty retained separately. The 100 demand-day IDs are resampled as common blocks across all loads/policies, and five learner repetitions retain their own level. No 1,500-independent-training-run assumption is made.','',
              'M3→M4 remains the original same-PPO architecture comparison. M4→M5 is a separately budgeted, explicitly limited tuning experiment. A failed or successful S1 result does not alter the original M4 claim. Test is now consumed; no model or schedule was revised after observing it. Fixed-map demand-day generalization does not establish new-map or real-world performance.','']
    (ROOT/'results/m5_summary.md').write_text('\n'.join(lines))
    # Full S1 trajectories against the same learner subset of S0.
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');colors=('#225E9A','#AF7621','#42866A','#9A4D8B','#C66144')
    for ax,load in zip(axes.flat,(*('low','medium','high'),'equal')):
        def rate(point):return point['mean_on_time_rate'] if load=='equal' else point['by_load'][load]['on_time_rate']
        curves=[]
        for color,a in zip(colors,tuning['audits']):
            x=[r['steps']/1000 for r in a['checkpoints']][1:];y=[rate(r) for r in a['checkpoints']][1:];curves.append(y)
            ax.plot(x,y,color=color,lw=1,alpha=.65,label=f"S1 seed {a['seed']}")
        ax.plot(x,np.mean(curves,axis=0),color='#143C62',lw=2.6,label='S1 mean')
        old=np.mean([[rate(r) for r in s0['curves'][str(s)]][1:] for s in seeds],axis=0)
        ax.plot(x,old,color='#AD650D',ls='--',lw=2.4,label='S0 same-seed mean')
        ax.axvline(512,color='#555555',ls=':',lw=1,label='Decay starts')
        ax.axvspan(819.2,1024,color='#777777',alpha=.07)
        ax.set(title=load.title(),xlabel='Training transitions (thousands)',ylabel='Dev-main mean daily on-time rate');ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
    fig.suptitle(f'M5 S1: {len(seeds)} fixed-budget learners vs the corresponding M4 repetitions\nOne LR change; current-checkpoint curves, not running best',fontsize=13)
    for ext in ('png','svg'):fig.savefig(ROOT/f'results/m5_tuning_curves.{ext}',dpi=180)
    plt.close(fig)

if __name__=='__main__':main()
