"""Paired product bootstrap over learner repetitions and matched demand-day IDs.

The day resample is shared across all loads and all methods. Learner identifiers
are resampled as paired blocks across learned methods, without claiming identical
trajectories. Conditional seed-only/day-only intervals remain separate from joint.
"""

import numpy as np


def crossed_bootstrap(left, right, repetitions=10000, seed=20261026):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.ndim != 3 or right.ndim != 3 or left.shape[1:] != right.shape[1:] or left.shape[1] != 3:
        raise ValueError('arrays must be learner x load x matched demand day')
    n = max(left.shape[0], right.shape[0])
    if left.shape[0] not in (1, n) or right.shape[0] not in (1, n):
        raise ValueError('learner repetitions must match, or one side must be a fixed baseline')
    delta = np.broadcast_to(left, (n, *left.shape[1:])) - np.broadcast_to(right, (n, *right.shape[1:]))
    delta = np.concatenate((delta, delta.mean(axis=1, keepdims=True)), axis=1)
    rng = np.random.default_rng(seed)
    learner_weights = rng.multinomial(n, np.full(n, 1 / n), size=repetitions) / n
    d = delta.shape[-1]
    day_weights = rng.multinomial(d, np.full(d, 1 / d), size=repetitions) / d
    seed_only = np.einsum('br,rl->bl', learner_weights, delta.mean(axis=2), optimize=True)
    day_only = np.einsum('ld,bd->bl', delta.mean(axis=0), day_weights, optimize=True)
    joint = np.einsum('br,rld,bd->bl', learner_weights, delta, day_weights, optimize=True)
    result = {}
    for i, load in enumerate(('low','medium','high','equal')):
        per_seed = delta[:, i].mean(-1)
        result[load] = {
            'estimate': float(per_seed.mean()),
            'learner_sample_sd': float(per_seed.std(ddof=1)) if n > 1 else None,
            'learner_mean_values': per_seed.tolist(),
            'seed_only_95_percentile': np.quantile(seed_only[:, i], [.025,.975]).tolist(),
            'day_only_95_percentile': np.quantile(day_only[:, i], [.025,.975]).tolist(),
            'joint_95_percentile': np.quantile(joint[:, i], [.025,.975]).tolist(),
        }
    return {'loads': result, 'repetitions': repetitions, 'rng_seed': seed,
            'learners': n, 'demand_day_ids': d,
            'method': 'paired crossed learner-ID/day-ID nonparametric product bootstrap; equal load weight',
            'limitations': 'five training repetitions limit bootstrap reliability; fixed map; no 1500-independent-episode assumption'}
