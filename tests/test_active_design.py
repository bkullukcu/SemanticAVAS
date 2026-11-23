"""Synthetic tests for source-aware OOD, acquisition and interval rules."""
import numpy as np
import pytest
from semanticavas.active_design import (
    fit_feature_domain,propose_listening_batch,group_conformal_radius,
    sample_weight_from_sem)


def test_outlying_sound_is_flagged():
    rng=np.random.default_rng(3)
    x=rng.normal(size=(12,4))
    groups=np.repeat(list('abcd'),3)
    reference=fit_feature_domain(x,groups=groups)
    score=reference.assess(np.vstack([x[0],np.full(4,20.)]))
    assert score.out_of_domain_warning.iloc[1]
    assert np.isfinite(score.nearest_training_distance).all()


def test_reserved_sources_never_selected():
    candidates=np.array([[10.],[9.],[8.],[7.],[6.],[5.]])
    ids=[f'S{i}' for i in range(6)]
    batch=propose_listening_batch(candidates,ids,np.array([[0.],[1.],[2.]]),
        n_select=2,candidate_groups=['reserved','reserved','A','B','C','C'],
        reserved_groups=['reserved'],max_per_group=1)
    assert not (batch.source_group=='reserved').any()
    assert batch.source_group.nunique()==2
    assert batch.iloc[0]['stimulus']=='S2'


def test_group_conformal_requires_enough_sources():
    observed=np.arange(6.).reshape(-1,1)
    pred=observed+1.
    with pytest.raises(ValueError,match='Too few independent'):
        group_conformal_radius(observed,pred,list('abcdef'),alpha=.10)
    y=np.arange(12.).reshape(-1,1)
    rad,meta=group_conformal_radius(y,y+1.,[str(i) for i in range(12)],alpha=.10)
    assert rad[0]==pytest.approx(1.)
    assert meta['n_calibration_groups']==12


def test_sem_weighting_and_validation():
    w=sample_weight_from_sem([1,1,5,10])
    assert w.mean()==pytest.approx(1.)
    assert w[0]>w[-1]
    with pytest.raises(ValueError):
        sample_weight_from_sem([-1,1])
