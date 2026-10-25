"""Chalmers and ARAUS adapters: deliberately separate perceptual endpoints.

Localization error (degrees) is not the same outcome as subjective truck-like
or annoyance; soundscape 1--5 ratings are not interchangeable with AVAS 0--100.
Original recordings and participants are NOT redistributed by this package.
"""
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import re
import numpy as np
import pandas as pd
from .acoustic_features import features_from_wav_bytes,FEATURE_NAMES

LOC_ENDPOINTS={'le':'localization_error_deg','lt':'localization_time_s','fl':'failure_rate_percent'}
ARAUS_SCALES=['pleasant','eventful','chaotic','vibrant','uneventful','calm','annoying','monotonous']


def load_chalmers_localization(path,*,processed=True):
    """Read Zenodo 14261300 processedData.xlsx (wide participant response table).
    Parses le_*, lt_*, fl_* columns and preserves participant-level rows.
    """
    if not processed:
        raise ValueError('rawData.xlsx is trial-level and requires separate trial order mapping; use processedData.xlsx')
    df=pd.read_excel(path)
    endpoints=[c for c in df if re.match(r'^(le|lt|fl)_[a-z0-9_]+$',str(c),re.I)]
    if not endpoints:raise ValueError('Expected le_*, lt_*, fl_* processed response columns')
    participant=next((c for c in df if str(c).lower() in ['participant','participant_id','id']),None)
    if participant is None:df.insert(0,'participant_id',np.arange(len(df)).astype(str));participant='participant_id'
    long=df.melt(id_vars=[participant],value_vars=endpoints,var_name='metric_condition',value_name='value').dropna(subset=['value'])
    long['endpoint_code']=long['metric_condition'].str.split('_',n=1).str[0].str.lower()
    long['endpoint']=long['endpoint_code'].map(LOC_ENDPOINTS)
    long['condition']=long['metric_condition'].str.split('_',n=1).str[1]
    long=long.rename(columns={participant:'participant'})
    long['value']=pd.to_numeric(long['value'],errors='coerce')
    long=long.dropna(subset=['value'])
    if (long['value']<0).any():raise ValueError('Negative localization outcome')
    return long[['participant','condition','endpoint','value']]


def localization_descriptives(long):
    if not {'participant','condition','endpoint','value'}.issubset(long):raise ValueError('normalized localization table required')
    return long.groupby(['condition','endpoint'],as_index=False).agg(n=('value','size'),median=('value','median'),q25=('value',lambda x:x.quantile(.25)),q75=('value',lambda x:x.quantile(.75)))


def load_araus_responses(path,*,allow_test=True):
    """Read v1/v2 `responses.csv` retaining source fold and original 1--5 scale.
    Standard clean split excludes attention/practice and keeps fold_r>=0.
    """
    data=pd.read_csv(path)
    required={'participant','fold_r','soundscape','masker','annoying','pleasant'}
    if not required.issubset(data.columns):raise ValueError('ARAUS schema mismatch: '+str(sorted(required-set(data.columns))))
    if 'is_attention' in data:data=data[data['is_attention'].fillna(0)==0]
    data=data[pd.to_numeric(data['fold_r'],errors='coerce')>=0].copy()
    if not allow_test:data=data[data['fold_r']!=0]
    labels=[c for c in ARAUS_SCALES if c in data]
    for col in labels:
        data[col]=pd.to_numeric(data[col],errors='coerce')
        if data[col].notna().any() and not data[col].dropna().between(1,5).all():raise ValueError(f'ARAUS {col} should have 1-5 ordinal values')
    data['domain']='urban_soundscape_not_individual_AVAS'
    data['scale']='original_1_to_5_ordinal'
    return data


def _araus_folds(data):
    """Original fixed benchmark: 1-5 CV, 0 and 6/7 reserved test, -1 excluded.
    Returning training indices never includes any reserved test data.
    """
    present=set(map(int,data['fold_r'].unique()))
    cv=sorted(present.intersection({1,2,3,4,5}))
    if len(cv)<2:raise ValueError('Insufficient non-test ARAUS folds')
    yield from ((f,[x for x in cv if x!=f],'cross_validation') for f in cv)
    for f in sorted(present.intersection({0,6,7})):
        yield f,cv,'independent_test'


def araus_leave_fold_out_baseline(data, target='annoying'):
    """Respect ARAUS official train/test split; no reserved-test contamination."""
    if target not in ARAUS_SCALES or target not in data:raise ValueError('invalid ARAUS target')
    rows=[]
    for fold,training_folds,phase in _araus_folds(data):
        train=data[data['fold_r'].isin(training_folds)][target].dropna()
        test=data[data['fold_r']==fold][target].dropna()
        if len(train)<2 or len(test)<1:continue
        mean=float(train.mean())
        rows.append({'heldout_fold':int(fold),'phase':phase,'n_test':len(test),
                     'mean_baseline_MAE':float((test-mean).abs().mean()),
                     'mean_training_rating':mean})
    return pd.DataFrame(rows)

def inspect_evat_wav_archive(path,*,max_files=None):
    """Zenodo 10610491 audio_examples.zip.
    Returns digital spectral descriptors only, NOT calibrated SPL/psychoacoustic
    loudness. The acoustic-to-perceptual label mapping is absent in this archive.
    """
    records=[]
    with ZipFile(path) as z:
        entries=[e for e in z.infolist() if not e.is_dir() and e.filename.lower().endswith('.wav')]
        for e in entries[:max_files]:
            features,metadata=features_from_wav_bytes(z.read(e))
            records.append({'path':e.filename,'filename':Path(e.filename).name,**dict(zip(FEATURE_NAMES,features)),**metadata})
    if not records:raise ValueError('No WAV audio examples in EVAT archive')
    return pd.DataFrame(records)


def araus_foldwise_ridge(data, *,target='annoying',feature_columns=('Leq_L_r','Leq_R_r','Navg_r','Savg_r'),alpha=10.0):
    """Train-only ridge, respecting fixed ARAUS CV and independent test sets.
    Original 1--5 ARAUS ordinal endpoints and domain metadata are preserved.
    """
    cols=[c for c in feature_columns if c in data]
    if len(cols)<2:raise ValueError('At least two measured ARAUS descriptor fields required')
    if target not in ARAUS_SCALES or target not in data:raise ValueError('invalid target')
    if alpha<=0:raise ValueError('alpha must be positive')
    rows=[]
    for fold,training_folds,phase in _araus_folds(data):
        tr=data[data['fold_r'].isin(training_folds)].copy();te=data[data['fold_r']==fold].copy()
        tr=tr[pd.to_numeric(tr[target],errors='coerce').notna()]
        te=te[pd.to_numeric(te[target],errors='coerce').notna()]
        if len(tr)<len(cols)+2 or len(te)<1:continue
        a=tr[cols].apply(pd.to_numeric,errors='coerce').to_numpy(float)
        b=te[cols].apply(pd.to_numeric,errors='coerce').to_numpy(float)
        med=np.nanmedian(a,axis=0)
        med[~np.isfinite(med)]=0
        a=np.where(np.isfinite(a),a,med);b=np.where(np.isfinite(b),b,med)
        mu=a.mean(axis=0);sd=a.std(axis=0);sd[sd<=1e-12]=1
        x=(a-mu)/sd; xt=(b-mu)/sd
        y=tr[target].astype(float).to_numpy();yt=te[target].astype(float).to_numpy()
        yc=y.mean()
        beta=np.linalg.solve(x.T@x+alpha*np.eye(x.shape[1]),x.T@(y-yc))
        pred=np.clip(xt@beta+yc,1,5)
        rows.append({'fold':int(fold),'phase':phase,'n_train':len(y),'n_test':len(yt),
                     'MAE_ridge':float(np.mean(abs(yt-pred))),
                     'MAE_train_mean':float(np.mean(abs(yt-yc))),
                     'target_unit':'ARAUS ordinal 1--5'})
    return pd.DataFrame(rows)
