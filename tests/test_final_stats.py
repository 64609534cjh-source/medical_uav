"""The final uncertainty calculation must preserve pairing and both levels."""
import numpy as np
from medical_uav.evaluation.final_stats import crossed_bootstrap


def test_exact_pairing_and_constant_shift():
    data=np.random.default_rng(3).random((5,3,20))
    zero=crossed_bootstrap(data,data,1000)
    shifted=crossed_bootstrap(data+.1,data,1000)
    for load in zero['loads']:
        assert np.max(np.abs(zero['loads'][load]['joint_95_percentile']))==0
        np.testing.assert_allclose(shifted['loads'][load]['joint_95_percentile'],[.1,.1],atol=1e-12)


def test_seed_and_day_uncertainty_are_not_pooled():
    varying_day=np.broadcast_to(np.linspace(0,1,20),(5,3,20)).copy()
    fixed=np.zeros((1,3,20))
    a=crossed_bootstrap(varying_day,fixed,1000)['loads']['equal']
    np.testing.assert_allclose(a['seed_only_95_percentile'],[.5,.5])
    assert a['day_only_95_percentile'][1]>a['day_only_95_percentile'][0]
    varying_seed=np.broadcast_to(np.arange(5)[:,None,None]/5,(5,3,20)).copy()
    b=crossed_bootstrap(varying_seed,fixed,1000)['loads']['equal']
    np.testing.assert_allclose(b['day_only_95_percentile'],[.4,.4])
    assert b['seed_only_95_percentile'][1]>b['seed_only_95_percentile'][0]
