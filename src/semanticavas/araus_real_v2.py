"""Leakage-guarded ARAUSv2 real-data perceptual benchmarks.

ARAUSv2 has independent evaluation folds (0,6,7), cross-validation
folds (1..5), and repeated common quality-control stimuli (fold -1).
No AVAS-specific construct equivalence is asserted.

Reference: Ooi et al., ARAUS and ARAUSv2.
Original ARAUS response values are ordinal ratings from 1 to 5.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGETS = ('annoying','pleasant','eventful','calm')
CV_FOLDS=(1,2,3,4,5)
TEST_FOLDS=(0,6,7)
FEATURES=(
 'smr','LAavg_r','LCavg_r','LAmax_r','LA10_r','LA90_r',
 'Navg_r','Nrmc_r','Savg_r','Favg_r','Ravg_r','Tavg_r',
 'Leq_L_r','Leq_R_r',
)
CATEGORIES=('bird','water','construction','traffic','wind','silence')


def load_araus_v2(responses,maskers,soundscapes=None):
    """Read published ARAUSv2 CSVs and verify key joins and folds.

    Does not load ARAUSv1 simultaneously (which would double-count records).
    No participant metadata is returned beyond the unmodified response table.
    """
    df=pd.read_csv(responses);mk=pd.read_csv(maskers)
    req=set(FEATURES+TARGETS+('fold_r','participant','soundscape','masker'))
    if missing:=req-set(df.columns):raise ValueError('Missing response columns '+str(sorted(missing)))
    if not {'masker','class','fold_m'}.issubset(mk):raise ValueError('Masker metadata schema mismatch')
    if mk.masker.duplicated().any():raise ValueError('Duplicate masker keys')
    df=df.merge(mk[['masker','class','fold_m']],how='left',on='masker',validate='many_to_one',indicator=True)
    if (df['_merge']!='both').any():raise ValueError('Some masker IDs do not match metadata')
    df.drop(columns=['_merge'],inplace=True)
    if soundscapes is not None:
        ss=pd.read_csv(soundscapes)
        if ss.soundscape.duplicated().any():raise ValueError('Duplicate soundscape keys')
        df=df.merge(ss[['soundscape','fold_s']],how='left',on='soundscape',validate='many_to_one')
        if df.fold_s.isna().any():raise ValueError('Unmatched soundscape metadata')
    for target in TARGETS:
        if not df[target].dropna().between(1,5).all():raise ValueError('ARAUS ordinal rating range violated')
    # Fold -1 includes common attention/practice/consistency stimuli in many splits.
    df=df[df.fold_r.isin(CV_FOLDS+TEST_FOLDS)].copy()
    if df.empty:raise ValueError('No official train / reserved test responses')
    if df[['participant','fold_r']].isna().any().any():raise ValueError('Invalid participant/fold')
    tr=df[df.fold_r.isin(CV_FOLDS)]
    for f in TEST_FOLDS:
        te=df[df.fold_r==f]
        if len(te):
            if set(tr.participant)&set(te.participant):raise ValueError('Participant leakage into independent test fold')
            if set(tr.masker)&set(te.masker):raise ValueError('Masker leakage into independent test fold')
            if set(tr.soundscape)&set(te.soundscape):raise ValueError('Soundscape leakage into independent test fold')
    return df


def _make_feature_table(x):
    a=x.loc[:,FEATURES+('class',)].copy()
    # Data already contains post-mixing acoustic and psychoacoustic descriptors.
    a['class']=a['class'].astype(str)
    return a


def _ridge():
    prep=ColumnTransformer([
        ('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),list(FEATURES)),
        ('category',OneHotEncoder(handle_unknown='ignore'),['class']),
    ])
    return Pipeline([('preprocess',prep),('regressor',Ridge(alpha=100.0))])


def _boost():
    # Use fixed published class definitions, no test statistics to learn categories.
    return HistGradientBoostingRegressor(loss='squared_error',max_iter=140,
        max_leaf_nodes=15,learning_rate=.055,min_samples_leaf=65,
        l2_regularization=5.0,random_state=2026)


def _boost_features(x):
    a=x.loc[:,FEATURES].apply(pd.to_numeric,errors='coerce').copy()
    # The measured descriptor fields are complete in published ARAUSv2; do not impute from test.
    if a.isna().any().any():raise ValueError('Missing numeric ARAUS descriptor requires train-only imputation')
    for cat in CATEGORIES:a['class_'+cat]=(x['class']==cat).to_numpy(dtype=int)
    return a


def _group_mean_predict(training,testing,target):
    grp=training.groupby(['class','smr'])[target].agg(['mean','size'])
    cls=training.groupby('class')[target].mean()
    mu=float(training[target].mean())
    result=[]
    for cat,smr in zip(testing['class'],testing.smr):
        if (cat,smr) in grp.index:
            item=grp.loc[(cat,smr)]
            # smooth small groups toward training-only class mean
            result.append((item['size']*item['mean']+25*float(cls.loc[cat]))/(item['size']+25))
        else:result.append(float(cls.get(cat,mu)))
    return np.asarray(result,float)


def predict_train_only(train,test,target,model_name):
    if model_name=='train_mean':return np.full(len(test),float(train[target].mean()))
    if model_name=='class_smr':return np.clip(_group_mean_predict(train,test,target),1,5)
    if model_name=='acoustic_ridge':
        model=_ridge();model.fit(_make_feature_table(train),train[target]);pred=model.predict(_make_feature_table(test))
    elif model_name=='acoustic_boost':
        model=_boost();model.fit(_boost_features(train),train[target]);pred=model.predict(_boost_features(test))
    else:raise ValueError('Unknown model')
    return np.clip(np.asarray(pred,float),1,5)


def evaluate_araus_v2(data,targets=TARGETS,models=('train_mean','class_smr','acoustic_ridge','acoustic_boost'), full_cv=False):
    """Validation = train 1-4/test 5; refit 1-5, untouched holdouts 0/6/7.

    All model settings fixed before independent folds are examined. Per-fold
    metrics are participant-response MAEs, not independent participant CIs.
    """
    results=[]
    for target in targets:
        for phase,training_folds,eval_folds in ([('cross_validation', tuple(f for f in CV_FOLDS if f != held), (held,)) for held in CV_FOLDS] if full_cv else [('development',(1,2,3,4),(5,))]) + [('independent_test',CV_FOLDS,TEST_FOLDS)]:
            train=data[data.fold_r.isin(training_folds)]
            for f in eval_folds:
                test=data[data.fold_r==f]
                if test.empty:continue
                y=test[target].to_numpy(float)
                for name in models:
                    pred=predict_train_only(train,test,target,name)
                    results.append({'target':target,'phase':phase,'fold':f,'model':name,
                                    'n_train':len(train),'n_test':len(test),
                                    'mae':float(np.mean(np.abs(y-pred))),
                                    'rmse':float(np.sqrt(np.mean((y-pred)**2))),
                                    'bias':float(np.mean(pred-y))})
    return pd.DataFrame(results)


def check_masker_audio_archives(maskers_csv,archives):
    """Inventory ZIP files including nested `maskersv2.zip` inside `dataverse_files.zip`.

    This checks filenames, not correctness of the calibrated audio signal.
    """
    import io,zipfile,collections
    all_names=collections.defaultdict(list)
    for p in map(Path,archives):
        with zipfile.ZipFile(p) as z:
            if 'maskersv2.zip' in z.namelist():
                with zipfile.ZipFile(io.BytesIO(z.read('maskersv2.zip'))) as inner:
                    members=inner.namelist()
            else:members=z.namelist()
            for s in members:
                if s.lower().endswith('.wav'):
                    all_names[Path(s).name].append(p.name)
    expected=set(pd.read_csv(maskers_csv).masker)
    return {'present':len(expected&set(all_names)), 'expected':len(expected),
            'missing':sorted(expected-set(all_names)),
            'duplicates':{k:v for k,v in all_names.items() if len(v)>1},
            'unexpected':sorted(set(all_names)-expected)}
