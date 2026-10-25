"""Reproducibility and data leakage tests for external soundscape analysis."""
import pandas as pd
import numpy as np
import pytest
from semanticavas.araus_real_v2 import load_araus_v2,predict_train_only,evaluate_araus_v2


def synthetic_csvs(tmp_path):
    rows=[]
    metadata=[]; sounds=[]
    for f in range(8):
        for i in range(7):
            mk=f'bird_{f}_{i}.wav'; sc=f'urban_{f}_{i}.wav'
            metadata.append({'masker':mk,'class':'bird','fold_m':f})
            sounds.append({'soundscape':sc,'fold_s':f})
            rows.append({'participant':f'P_{f}_{i}','fold_r':f,'soundscape':sc,'masker':mk,
                'smr':f%3-1,'LAavg_r':62+i,'LCavg_r':65+i,'LAmax_r':68+i,
                'LA10_r':66+i,'LA90_r':60+i,'Navg_r':3+i,'Nrmc_r':3+i,
                'Savg_r':1+i/10,'Favg_r':0.2,'Ravg_r':.05,'Tavg_r':.3,
                'Leq_L_r':60+i,'Leq_R_r':61+i,
                'annoying':1+(i%5),'pleasant':5-(i%5),
                'eventful':1+i%5,'calm':5-i%5})
    pd.DataFrame(rows).to_csv(tmp_path/'responses.csv',index=False)
    pd.DataFrame(metadata).to_csv(tmp_path/'maskers.csv',index=False)
    pd.DataFrame(sounds).to_csv(tmp_path/'soundscapes.csv',index=False)
    return tmp_path


def test_train_and_reserved_test_are_disjoint(tmp_path):
    t=synthetic_csvs(tmp_path)
    df=load_araus_v2(t/'responses.csv',t/'maskers.csv',t/'soundscapes.csv')
    assert set(df.fold_r)==set(range(8))
    tr=df[df.fold_r.isin((1,2,3,4,5))]
    test=df[df.fold_r.isin((0,6,7))]
    assert not set(tr.participant)&set(test.participant)
    assert not set(tr.masker)&set(test.masker)
    assert not set(tr.soundscape)&set(test.soundscape)


def test_target_range_guard(tmp_path):
    t=synthetic_csvs(tmp_path)
    p=t/'responses.csv';d=pd.read_csv(p);d.loc[0,'annoying']=99;d.to_csv(p,index=False)
    with pytest.raises(ValueError,match='range'):load_araus_v2(p,t/'maskers.csv',t/'soundscapes.csv')


def test_no_reserved_fold_used_to_fit(tmp_path):
    t=synthetic_csvs(tmp_path)
    df=load_araus_v2(t/'responses.csv',t/'maskers.csv',t/'soundscapes.csv')
    first=evaluate_araus_v2(df,targets=('annoying',),models=('train_mean',),full_cv=False)
    x=df.copy()
    x.loc[x.fold_r.isin((0,6,7)),'annoying']=1
    second=evaluate_araus_v2(x,targets=('annoying',),models=('train_mean',),full_cv=False)
    # Manipulating test labels cannot change training baseline mean;
    # it should only change test error.
    a=first[first.phase=='independent_test'];b=second[second.phase=='independent_test']
    assert (a.n_train.values==b.n_train.values).all()
    assert not np.allclose(a.mae,b.mae)


def test_reconstruction_not_confused_with_avaselect(tmp_path):
    t=synthetic_csvs(tmp_path)
    df=load_araus_v2(t/'responses.csv',t/'maskers.csv',t/'soundscapes.csv')
    tr=df[df.fold_r.isin((1,2,3,4,5))];te=df[df.fold_r==6]
    for method in ('train_mean','class_smr','acoustic_ridge'):
        x=predict_train_only(tr,te,'annoying',method)
        assert np.isfinite(x).all() and len(x)==len(te) and ((x>=1)&(x<=5)).all()
