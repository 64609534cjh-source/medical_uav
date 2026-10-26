"""Assemble the completed Gen2 evidence without changing frozen statistics/models."""
import json
from pathlib import Path
import numpy as np
from medical_uav.training.io import write_json


def read(p):return json.loads(Path(p).read_text())


def main():
    dev=read('results/gen2_development.json');test=read('results/gen2_test_v2.json');lat=read('results/gen2_latency.json');audit=read('results/gen2_training_audit.json')
    methods=['m5','destination','relational'];deltas={m:test['comparisons'][m+' - m5']['loads']['equal'] for m in methods[1:]}
    out=['# Gen2 — Destination-conditioned DELIVER and relational attention','',
         'The destination-conditioned head improves equal-load Test-v2 on-time rate by **+0.88 pp** over frozen M5 (pointwise paired 95% interval **[+0.41, +1.37] pp**), with all five paired learner differences positive. The tested single-layer relational model has no clear overall gain (**−0.05 pp**, **[−0.53, +0.44] pp**) and higher inference cost. These are results for the specified implementations and budget, not a general ranking of all attention models or evidence of theoretical optimality.',
         '', 'Retain M5 as the first-generation formal reference and the destination head as the supported second-generation upgrade. The tested attention model does not justify replacing M5. No learner seed is selected using Test-v2.', '', '## Scope','','Generation 1 (`3c05ad3`) remains frozen. This stage tests two separate architecture changes, preserving the 403-dimensional information, 26 actions, K=12, simulator, reward, PPO and M5 late learning-rate schedule. Each new method has five scratch learners at 1,024,000 transitions. No combined network, expanded candidates or hyperparameter search was added.','', '## New Test-v2: selected-best checkpoints','','| Method | Low (%) | Medium (%) | High (%) | Equal (%) |','|---|---:|---:|---:|---:|']
    for m in methods+['slack']:
        r=test['methods'][m];vals=[f'{v*100:.2f} ± {r["sample_sd"][i]*100:.2f}' if r['sample_sd'] is not None else f'{v*100:.2f}' for i,v in enumerate(r['mean'])]
        out.append('| '+m+' | '+' | '.join(vals)+' |')
    out+=['','± is the sample SD across five learner seeds. The 100 new demand IDs 300000–300099 were sealed before development; all 15 models and statistical code were frozen before testing.','', '| Primary contrast | Equal delta (pp) | Paired joint 95% interval (pp) | Positive learner pairs |','|---|---:|---:|---:|']
    for m,d in deltas.items():
        ci=d['joint_95_percentile'];positive=sum(x>0 for x in d['learner_mean_values'])
        out.append(f'| {m} − m5 | {100*d["estimate"]:+.2f} | [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}] | {positive}/5 |')
    out+=['','The destination differences are −0.06 pp at Low, +0.94 pp at Medium and +1.78 pp at High; the corresponding pointwise paired intervals are [−0.49,+0.27], [+0.08,+1.72] and [+0.86,+2.64] pp. Low does not show a clear advantage. These new demand days differ from the first-generation Test: the identical frozen M5 weights score 85.02% here versus 86.42% on the old Test. Cross-cohort score subtraction is not a measure of training degradation.', '', 'Intervals resample learner-ID and matched demand-day-ID blocks; common day weights span loads. These are pointwise exploratory intervals without multiplicity correction. Five learners and one fixed map limit broader claims. The full report includes separate learner/day components and all daily differences.','', '## Development: peak selection and retention','','| Method | Dev best | Dev last | Dev last-three mean | Dev-extra best | Dev-extra last |','|---|---:|---:|---:|---:|---:|']
    retention={}
    for m in methods:
        tail=np.array([r['on_time'] for r in dev['curves'][m]])[:,-3:,3].mean(1)
        values=[dev['dev_best']['methods'][m]['mean'][3],dev['dev_last']['methods'][m]['mean'][3],float(tail.mean()),dev['dev_extra_best']['methods'][m]['mean'][3],dev['dev_extra_last']['methods'][m]['mean'][3]]
        retention[m]={'dev_best':values[0],'dev_last':values[1],'dev_last_three_mean':values[2],'dev_extra_best':values[3],'dev_extra_last':values[4]}
        out.append('| '+m+' | '+' | '.join(f'{100*x:.2f}%' for x in values)+' |')
    out+=['','Last-three means use the fixed checkpoints at 819,200, 921,600 and 1,024,000 transitions. Dev-extra is development, not a second independent test. Test-v2 includes best only, so no Test claim about last-checkpoint stability is made.','', '## Compute cost','','| Method | Parameters | Batch 1 p50 / p95 (µs) | Batch 8 p50 / p95 (µs per batch) |','|---|---:|---:|---:|']
    for m in methods:
        d=lat['aggregate'][m];b1=d['by_batch']['1'];b8=d['by_batch']['8']
        out.append(f'| {m} | {d["parameters"]:,} | {b1["p50_us_per_batch"]:.1f} / {b1["p95_us_per_batch"]:.1f} | {b8["p50_us_per_batch"]:.1f} / {b8["p95_us_per_batch"]:.1f} |')
    out+=['','Canonical masked categorical argmax, including critic, measured with one CPU thread, identical real observation/mask archive, 200 warmups and 2,000 calls per model/batch. Table entries are medians of the five per-model quantiles; they are not pooled p95s. M5’s separately verified factored action-only inference remains available; this fair architecture comparison uses the canonical interface for all methods.','', '| New method | Run wall time range (s) | Sum of five run wall times (s) |','|---|---:|---:|']
    for m in methods[1:]:
        rs=[r['wall_seconds'] for r in audit['runs'] if r['name'].startswith(m)]
        out.append(f'| {m} | {min(rs):.1f}–{max(rs):.1f} | {sum(rs):.1f} |')
    out+=['','Runs use five concurrent single-thread learner processes. Summed per-run wall time is not the suite elapsed time; the manifest retains actual suite wall time.','', '## Top-12 exposure and type trade-offs','','| M5 load | States >12 pending | Hidden same-destination partner | Direct-on-time hidden partner |','|---|---:|---:|---:|']
    for l in ['low','medium','high']:
        r=dev['congestion']['methods']['m5'][l]
        out.append(f'| {l} | {100*r["over12_state_fraction"]:.2f}% | {100*r["hidden_partner_state_fraction"]:.2f}% | {100*r["hidden_direct_on_time_partner_state_fraction"]:.2f}% |')
    out+=['','These Dev-extra diagnostics use simulator details only after/beside policy inference. A hidden destination match is a possible batching opportunity, not a feasible route certificate. Congested days can already be harder, so correlations do not identify a causal K=12 loss. No K=24 or action-interface expansion was performed.','', 'On Test-v2, destination conditioning increases plasma daily rates by +3.53/+1.90 pp and supplies by +3.17/+5.78 pp at Medium/High. Medicine changes by −0.49/−0.22 pp; sample changes by −0.71/+0.72 pp. These subtype differences are descriptive, with no simultaneous-inference claim. The model does not improve every type or load. In Dev-extra, the mean daily batch-DELIVER fraction rises from 21.47% to 24.01%, and pooled expired-after-pick/picked falls from 0.56% to 0.28%; these are policy behavior associations, not causal module attribution.\n\nAll four order types and three loads are reported in the development and Test-v2 reports, with day denominators. An equal-load gain cannot be described as universal improvement when a type/load declines.','', '## Verification and artifacts','', f'- Ten complete runs: {audit["budget_total"]:,} transitions, 10,000 PPO updates, 110 Dev checkpoints, no illegal action or nonfinite required training diagnostic/checkpoint parameter.', '- Each source archive was verified against its manifest; configs, selection rules, complete days, model hashes and first-generation file hashes were audited.', '- Training source commit: `'+', '.join(audit['formal_source_commits'])+'`.',f'- New Test-v2: {test["episodes"]:,} full episodes, {test["decisions"]:,} decisions, no illegal actions.', '- [Development detail](gen2_development.md), [Test-v2 detail](gen2_test_v2.md), [learning curves](gen2_learning_curves.png), [focused tail curves](gen2_tail_curves.png), [Test-v2 paired differences](gen2_test_v2_paired.png).', '- [Architecture and reproduction notes](../docs/gen2_architectures.md), [training protocol](../configs/gen2_protocol.json), [Test-v2 protocol](../configs/gen2_test_v2_protocol.json).', '- [Final integrity audit](gen2_final_audit.json) confirms 65 first-generation files, 15 original selected models, all 15 Test-v2 models and 55 frozen source/development files retain their recorded hashes.', '- Raw checkpoints, complete daily metrics and update logs are retained under `outputs/gen2_architectures`, `outputs/gen2_destination_devextra`, `outputs/gen2_relational_devextra`, `outputs/gen2_m5_devextra` and `outputs/gen2_test_v2`.', '', 'No method has been proved theoretically optimal. New-map generalization and an aligned optimality bound remain untested. Test-v2 is now consumed; no architecture or hyperparameter changes followed its results.']
    Path('results/gen2_summary.md').write_text('\n'.join(out)+'\n')
    write_json(Path('results/gen2_summary.json'),{'generation':2,'training_source_commits':audit['formal_source_commits'],'retention':retention,'test_v2_primary_differences':deltas,'latency':lat['aggregate'],'test_v2_status':'consumed','generation1_preserved':True})


if __name__=='__main__':main()
