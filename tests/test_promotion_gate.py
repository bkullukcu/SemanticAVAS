"""The published predictor is not replaced without independent evidence."""
import numpy as np
import pytest
from semanticavas.promotion_gate import assess_model_promotion


def test_old_six_source_dataset_cannot_promote_candidate():
    y=np.zeros((12,2));ref=np.ones_like(y)*4;new=np.ones_like(y)*2
    groups=np.repeat(list('abcdef'),2)
    report=assess_model_promotion(y,new,ref,groups,independent_test_certified=True)
    assert report['recommendation']=='retain_incumbent'
    assert 'Insufficient' in report['reason']


def test_uncertified_development_set_cannot_promote():
    y=np.zeros((12,2));ref=np.ones_like(y)*3
    candidate=np.ones_like(y)
    report=assess_model_promotion(y,candidate,ref,[str(i) for i in range(12)])
    assert report['recommendation']=='retain_incumbent'


def test_independent_balanced_consistent_gain_passes_gate():
    y=np.zeros((12,2));ref=np.ones_like(y)*3;candidate=np.ones_like(y)
    report=assess_model_promotion(
        y,candidate,ref,[str(i) for i in range(12)],
        independent_test_certified=True,n_boot=300)
    assert report['recommendation']=='eligible_for_author_assessment'
    assert report['paired_source_bootstrap_ci'][1]<0


def test_worse_candidate_fails():
    y=np.zeros((12,2));ref=np.ones_like(y);candidate=np.ones_like(y)*4
    report=assess_model_promotion(
        y,candidate,ref,[str(i) for i in range(12)],
        independent_test_certified=True,n_boot=300)
    assert report['recommendation']=='retain_incumbent'
