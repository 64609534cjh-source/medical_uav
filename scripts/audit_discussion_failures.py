"""R1: observational diagnosis on existing Dev-extra; no new Test execution."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
LOADS = ('low', 'medium', 'high')
TYPES = ('plasma', 'medicine', 'sample', 'supplies')
OUTCOMES = ('on_time', 'late', 'expired_unpicked', 'expired_after_pick')


def main():
    rows = {}
    for method in ('m5', 'destination'):
        rows[method] = [r for seed in range(1, 6)
                        for r in json.loads((ROOT / f'outputs/gen3_baseline_diagnostics/{method}_s{seed}.json').read_text())['episodes']]
        keys = {(r['learner_seed'], r['episode_load'], r['demand_seed']) for r in rows[method]}
        if len(keys) != 1500 or len(rows[method]) != 1500 or any(r['split'] != 'dev-extra' for r in rows[method]):
            raise ValueError('requires five learners × three loads × 100 Dev-extra days')
    result = {'scope': 'observational Dev-extra; five learner repetitions of 100 shared days; no causal attribution',
              'methods': {}, 'representative_selection': 'learner 1; fixed buckets; median absolute paired daily gap, then load/day tie-break',
              'representatives': []}
    lines = ['# R1: development failure audit', '', result['scope'], '',
             'Outcomes are mutually exclusive. Rates below average eligible day/type rates, '
             'then learners; zero-release type days are omitted. Counts repeat each day under five policies.', '',
             '| Method | Load | Type | Released* | Eligible days | On-time % | Late % | Expired unpicked % | Expired after pick % |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for method, records in rows.items():
        result['methods'][method] = {}
        for load in LOADS:
            selected = [r for r in records if r['episode_load'] == load]
            congestion = {k: sum(r['diagnostics']['congestion'][k] for r in selected)
                          for k in ('states', 'over_12_states', 'hidden_direct_on_time_matching_states')}
            section = {'congestion': congestion, 'types': {}}
            result['methods'][method][load] = section
            for kind in TYPES:
                outcomes = [r['diagnostics']['outcomes_by_type'][kind] for r in selected]
                if any(sum(x[o] for o in OUTCOMES) != x['released'] for x in outcomes):
                    raise ValueError('outcome partition mismatch')
                eligible = [x for x in outcomes if x['released']]
                summary = {'released_repeated': sum(x['released'] for x in outcomes),
                           'eligible_unique_days': len(eligible) // 5,
                           'mean_daily_rates': {o: statistics.mean(x[o]/x['released'] for x in eligible) for o in OUTCOMES},
                           'counts': {o: sum(x[o] for x in outcomes) for o in OUTCOMES},
                           'failed_never_visible': sum(r['diagnostics']['failure_diagnosis']['counts_by_type'][kind].get('failed_never_visible', 0) for r in selected)}
                section['types'][kind] = summary
                lines.append('| ' + ' | '.join([method, load, kind, str(summary['released_repeated']),
                             str(summary['eligible_unique_days'])] + [f'{100*summary["mean_daily_rates"][o]:.2f}' for o in OUTCOMES]) + ' |')
    lines += ['', '*Released counts include five repetitions of the same demand days.', '',
              '| Method | Load | Over-12 decision states (%) | Hidden direct-on-time matching states (%) | Never-visible failures / all failures (%) |',
              '|---|---|---:|---:|---:|']
    for method in rows:
        for load in LOADS:
            section = result['methods'][method][load]
            c = section['congestion']
            missed = sum(t['failed_never_visible'] for t in section['types'].values())
            failed = sum(sum(t['counts'][o] for o in OUTCOMES[1:]) for t in section['types'].values())
            lines.append(f'| {method} | {load} | {100*c["over_12_states"]/c["states"]:.3f} | '
                         f'{100*c["hidden_direct_on_time_matching_states"]/c["states"]:.3f} | {100*missed/failed:.3f} |')
    index = {m: {(r['episode_load'], r['demand_seed']): r for r in rs if r['learner_seed'] == 1} for m, rs in rows.items()}
    paired = []
    for key, row in index['destination'].items():
        base = index['m5'][key]
        delta = row['on_time_rate'] - base['on_time_rate']
        paired.append((key, delta, row, base))
    for label, predicate in [('success', lambda x: x[2]['on_time_rate'] == 1 and x[3]['on_time_rate'] == 1),
                             ('failure_both', lambda x: x[2]['on_time_rate'] < .9 and x[3]['on_time_rate'] < .9),
                             ('improvement', lambda x: x[1] > 0), ('regression', lambda x: x[1] < 0)]:
        selected = sorted([x for x in paired if predicate(x)], key=lambda x: (abs(x[1]), x[0]))
        if not selected:
            raise ValueError('missing representative bucket: ' + label)
        key, delta, row, base = selected[len(selected)//2]
        result['representatives'].append({'bucket': label, 'learner_seed': 1, 'load': key[0], 'day': key[1],
                                          'destination_rate': row['on_time_rate'], 'm5_rate': base['on_time_rate'],
                                          'delta_pp': 100*delta})
    lines += ['', '## Representative days', '',
              'These are descriptive cases selected from Dev by the fixed bucket rule, not effect-size evidence. '
              'Full paired decision timelines are in `discussion_traces.json.gz`, reproduced by `scripts/trace_discussion.py`.', '',
              '| Bucket | Learner | Load | Day | Destination % | M5 % | Delta pp |', '|---|---:|---|---:|---:|---:|---:|']
    for r in result['representatives']:
        lines.append(f'| {r["bucket"]} | 1 | {r["load"]} | {r["day"]} | {100*r["destination_rate"]:.2f} | {100*r["m5_rate"]:.2f} | {r["delta_pp"]:+.2f} |')
    lines += ['', '## Interpretation and next tests', '',
              'Candidate truncation is a testable interface hypothesis, not an established cause: report its actual prevalence above before prioritizing K=24. '
              'Never visible at decision times also includes orders that arrive and expire within a macro action; it is not synonymous with top-K exclusion.', '',
              'Short-horizon planning is a separate online-compute hypothesis. Action-underway deadline crossings cannot determine avoidability. '
              'Any planner must reconstruct only public state and sample independent future scenarios; diagnostic simulator details cannot enter its policy input.', '',
              'Prioritize a candidate-window feasibility/control test and a public-information lookahead feasibility test after these descriptive checks. '
              'Training settings, reward, type guardrails and checkpoint selection remain unchanged. '
              'R1 is partial until representative traces and new development registrations are reviewed; no new split is asserted unused.']
    (ROOT / 'results/discussion_failure_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    (ROOT / 'results/discussion_failure_audit.md').write_text('\n'.join(lines) + '\n')
    print('Wrote complete load/type/outcome table and four balanced representative-day selections')


if __name__ == '__main__':
    main()
