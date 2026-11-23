"""Decision rule for promoting a candidate semantic prediction model.

The only admissible evidence comes from an independent test collection
reserved before model/feature selection, with correct source grouping.
"""
from __future__ import annotations

import numpy as np


def assess_model_promotion(observed, candidate, incumbent, source_groups, *,
                           independent_test_certified=False,
                           min_groups=10, n_boot=10000,
                           max_scale_regression=2., seed=2026):
    """Return a conservative promotion recommendation and supporting estimates.

    All numeric inputs are aligned held-out predictions, not in-sample fits.
    Testing a model on the data used for feature selection disqualifies it.
    """
    y=np.asarray(observed,float)
    candidate=np.asarray(candidate,float)
    incumbent=np.asarray(incumbent,float)
    sources=np.asarray(source_groups,str)
    if (y.ndim!=2 or y.shape!=candidate.shape or y.shape!=incumbent.shape
            or len(sources)!=len(y) or len(y)==0):
        raise ValueError('Aligned, nonempty two-dimensional prediction matrices required')
    if not np.isfinite(y).all() or not np.isfinite(candidate).all() or not np.isfinite(incumbent).all():
        raise ValueError('All held-out values must be finite')
    if min_groups<3 or n_boot<100 or max_scale_regression<0:
        raise ValueError('Invalid evidence thresholds')
    unique=np.unique(sources)
    delta=np.abs(candidate-y)-np.abs(incumbent-y)
    cluster=np.array([delta[sources==source].mean(axis=0) for source in unique])
    group_mean=cluster.mean(axis=0)
    mean_delta=float(np.mean(group_mean))
    result={'independent_test_certified':bool(independent_test_certified),
            'source_groups':len(unique),'samples':len(y),
            'candidate_mae':float(np.abs(candidate-y).mean()),
            'incumbent_mae':float(np.abs(incumbent-y).mean()),
            'equal_source_delta_mae':mean_delta,
            'per_scale_equal_source_delta':group_mean.tolist(),
            'recommendation':'retain_incumbent',
            'reason':''}
    if not independent_test_certified:
        result['reason']='Development data cannot validate model promotion'
        return result
    if len(unique)<min_groups:
        result['reason']='Insufficient independent test source groups'
        return result
    rng=np.random.default_rng(seed)
    draw=rng.integers(0,len(unique),size=(n_boot,len(unique)))
    boot=cluster[draw].mean(axis=(1,2))
    lo,hi=np.quantile(boot,[.025,.975])
    result['paired_source_bootstrap_ci']=[float(lo),float(hi)]
    if hi>=0:
        result['reason']='No supported reduction in error across held-out sources'
    elif np.any(group_mean>max_scale_regression):
        result['reason']='At least one semantic scale regressed above the allowed margin'
    else:
        result['recommendation']='eligible_for_author_assessment'
        result['reason']='Candidate passes prespecified numerical gates; author approval still required'
    return result
