import io
import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile
from semanticavas.empirical import (validate_semantic_ratings,aggregate_semantics,bootstrap_profiles,
    rank_existing_sounds,bootstrap_target,leave_one_participant_out,pca_stimulus_profiles,
    nested_leave_one_sound_out)
from semanticavas.acoustic_features import features_from_wav_bytes,FEATURE_NAMES


def sample():
    return pd.DataFrame([{'participant':p,'stimulus':s,'truck_like':float(20+20*i+5*j),
                          'annoying':float(70-10*i+j)} for j,p in enumerate(['p1','p2','p3'])
                          for i,s in enumerate(['S01','S02','S03'])])


def test_aggregation_and_counts_without_imputation():
    x=sample();x.loc[(x.participant=='p2')&(x.stimulus=='S03'),'annoying']=np.nan
    m,n=aggregate_semantics(x,['truck_like','annoying'])
    assert n.loc['S03','annoying']==2
    assert m.loc['S03','annoying']==pytest.approx((50+52)/2)


def test_duplicate_rejection():
    x=sample();q=pd.concat([x,x.iloc[[0]]],ignore_index=True)
    with pytest.raises(ValueError,match='duplicate'):
        validate_semantic_ratings(q,['truck_like'])


def test_invalid_range_rejection():
    x=sample();x.loc[0,'annoying']=103
    with pytest.raises(ValueError,match='outside'):
        validate_semantic_ratings(x,['annoying'])


def test_bootstrap_determinism_and_missingness():
    x=sample();x.loc[0,'annoying']=np.nan
    a,b=bootstrap_profiles(x,['truck_like','annoying'],n_boot=100,seed=3)
    c,d=bootstrap_profiles(x,['truck_like','annoying'],n_boot=100,seed=3)
    pd.testing.assert_frame_equal(a,c);pd.testing.assert_frame_equal(b,d)
    assert a.shape==(3,2)


def test_ranking_and_selection():
    x=sample();m,_=aggregate_semantics(x,['truck_like','annoying'])
    z=rank_existing_sounds(m,{'truck_like':61,'annoying':50})
    assert len(z)==3
    boot=bootstrap_target(x,{'truck_like':61},n_boot=200,seed=3)
    assert np.isclose(boot.bootstrap_selection_fraction.sum(),1)
    loo=leave_one_participant_out(x,{'truck_like':61})
    assert len(loo)==3


def test_pca_scores():
    x=sample();m,_=aggregate_semantics(x,['truck_like','annoying'])
    scores,loads,expl=pca_stimulus_profiles(m,['truck_like','annoying'])
    assert scores.shape==(3,2) and loads.shape==(2,2)
    assert expl.sum()==pytest.approx(1.)


def test_nested_loo_baseline_is_training_only():
    x=np.arange(15,dtype=float)[:,None]
    y=np.linspace(10,90,15)[:,None]
    pred,base,alpha=nested_leave_one_sound_out(x,y)
    assert pred.shape==y.shape and np.isfinite(pred).all()
    assert base[0,0]==pytest.approx(y[1:].mean())
    assert len(alpha)==15


def test_audio_descriptor_shape_and_digital_units():
    fs=16000;t=np.arange(fs)/fs
    y=(.05*np.sin(2*np.pi*300*t)).astype('float32')
    buf=io.BytesIO();wavfile.write(buf,fs,y)
    features,meta=features_from_wav_bytes(buf.getvalue())
    assert len(features)==len(FEATURE_NAMES)==10
    assert np.isfinite(features).all()
    assert meta['channels']==1 and meta['sample_rate_hz']==fs


def test_grouped_cannot_leak_held_out_sibling():
    from semanticavas.empirical import nested_leave_one_group_out
    x=np.arange(15,dtype=float)[:,None]
    y=np.linspace(10,90,15)[:,None]
    groups=np.array(['a']*3+['b']*3+[f'x{i}' for i in range(9)])
    pred,base,alphas=nested_leave_one_group_out(x,y,groups)
    assert pred.shape==y.shape and np.isfinite(pred).all()
    assert np.allclose(base[:3,0],y[3:].mean())
    assert np.allclose(alphas[:3],alphas[0])
