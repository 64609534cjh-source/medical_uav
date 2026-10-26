"""Matched daily type rates with zero-release days omitted consistently.

The same learner/day resampling weights span loads and both methods. Missingness
is a property of demand only, verified common across learners and methods.
"""
import numpy as np


def type_bootstrap(left,right,repetitions=10000,seed=20261026):
    left,right=np.asarray(left,float),np.asarray(right,float)
    if left.shape!=right.shape or left.ndim!=3 or left.shape[1]!=3:raise ValueError('matching learner x 3 loads x days required')
    valid=np.isfinite(left[0])
    if not (np.array_equal(np.isfinite(left),np.broadcast_to(valid,left.shape)) and np.array_equal(np.isfinite(right),np.broadcast_to(valid,right.shape))):raise ValueError('demand-only missingness required')
    if not valid.any(-1).all():raise ValueError('each load requires eligible days')
    delta=np.where(valid,left-right,0);n,_,d=delta.shape
    rng=np.random.default_rng(seed);sw=rng.multinomial(n,np.full(n,1/n),size=repetitions)/n;dw=rng.multinomial(d,np.full(d,1/d),size=repetitions)/d
    den=dw@valid.T
    joint=np.divide(np.einsum('br,rld,bd->bl',sw,delta,dw,optimize=True),den,out=np.full((repetitions,3),np.nan),where=den>0)
    day=np.divide(delta.mean(0)@dw.T,den.T,out=np.full((3,repetitions),np.nan),where=den.T>0).T
    per=delta.sum(-1)/valid.sum(-1);seed_only=sw@per
    point=per.mean(0)
    output={}
    for i,load in enumerate(('low','medium','high')):
        output[load]={'estimate':float(point[i]),'learner_sample_sd':float(per[:,i].std(ddof=1)) if n>1 else None,'learner_mean_values':per[:,i].tolist(),'eligible_days':int(valid[i].sum()),'joint_95_percentile':np.nanquantile(joint[:,i],[.025,.975]).tolist(),'seed_only_95_percentile':np.quantile(seed_only[:,i],[.025,.975]).tolist(),'day_only_95_percentile':np.nanquantile(day[:,i],[.025,.975]).tolist(),'valid_joint_replicates':int(np.isfinite(joint[:,i]).sum())}
    return {'loads':output,'repetitions':repetitions,'rng_seed':seed,'method':'paired crossed learner/day product bootstrap; per-load weighted normalization on nonzero-release days','scope':'exploratory pointwise intervals, no multiplicity correction; five learners limit uncertainty estimates'}
