"""Reconstruct a two-session 35-stimulus semantic dataset from MATLAB archives.

Privacy: don't export subjects' original names/demographics. Stimulus rows in
MATLAB `vp.results` are in PRESENTATION order, NOT stimulus order. The MATLAB
`vp.expOrder` is a 1-indexed stimulus-index permutation and MUST be applied.
"""
from __future__ import annotations
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import hashlib
import numpy as np
import pandas as pd
from scipy.io import loadmat
from .acoustic_features import features_from_wav_bytes, FEATURE_NAMES

SCALES_1=('electric_like','truck_like','annoying')
SCALES_2=('buzzing','harsh','whistling','futuristic')
SCALES=SCALES_1+SCALES_2
MAT_PATHS=('Experiments/Pedestrian_Test_1.mat','Experiments/Pedestrian_Test_2.mat')
WAV_BASE='Sounds_Pedestrian_Listening_Test/'


def _load_exp(zf,path):
    return loadmat(BytesIO(zf.read(path)), squeeze_me=True, struct_as_record=False)['exp']


def _verify_soundlist(e1,e2):
    a=[str(s.name) for s in np.ravel(e1.stim.stimList)]
    b=[str(s.name) for s in np.ravel(e2.stim.stimList)]
    if a!=b or len(a)!=35 or len(set(a))!=35:
        raise ValueError('session stimulus lists do not match or do not contain 35 unique WAVs')
    return a


def _decode_trial_matrix(v, nscales):
    m=np.asarray(v.results,float)
    order=np.asarray(v.expOrder,int).reshape(-1)
    if m.shape!=(35,nscales) or len(order)!=35 or sorted(order.tolist())!=list(range(1,36)):
        raise ValueError('invalid trial ordering or number of ratings')
    out=np.full_like(m,np.nan)
    out[order-1,:]=m
    if np.isinf(out).any() or np.nanmin(out)<0 or np.nanmax(out)>100:
        raise ValueError('out-of-range/nonfinite values in rating matrix')
    return out


def load_35_stimulus_ratings(archive):
    """Return anonymous, one-row-per-subject-and-stimulus ratings and ID mapping.

    Names and demographic variables are never exported. Retains absent ratings as
    NaN. Explicitly excludes the nonparticipant `Test` entry in the original MAT.
    """
    with ZipFile(archive) as z:
        e1,e2=[_load_exp(z,p) for p in MAT_PATHS]
        names=_verify_soundlist(e1,e2)
        v1=list(np.ravel(e1.vp));v2=list(np.ravel(e2.vp))
        if len(v1)!=21 or len(v2)!=21:
            raise ValueError('expected 20 subjects plus one test record')
        if str(v1[0].name).lower()!='test' or str(v2[0].name).lower()!='test':
            raise ValueError('archive test entry cannot be identified')
        if [str(v.name) for v in v1[1:]] != [str(v.name) for v in v2[1:]]:
            raise ValueError('sessions have different subject IDs/orders')
        rows=[]
        for idx,(a,b) in enumerate(zip(v1[1:],v2[1:]),1):
            r1=_decode_trial_matrix(a,3)
            r2=_decode_trial_matrix(b,4)
            merged=np.concatenate([r1,r2],axis=1)
            for j,sound in enumerate(names):
                rows.append(dict(participant=f'P{idx:02d}', stimulus=f'S{j+1:02d}', stimulus_filename=sound,
                                 **{k:float(v) for k,v in zip(SCALES,merged[j])}))
        f=pd.DataFrame(rows)
        if f.shape[0]!=700 or f[list(SCALES)].notna().sum().sum()<4800:
            raise ValueError('unexpected participant/grid coverage')
        mapping=pd.DataFrame([{'stimulus':f'S{i+1:02d}','stimulus_filename':name,
                               'archive_class':'numbered_candidate' if i<26 else 'named_reference',
                               'published_synthesis_table':'reported' if i<18 else 'not_reported'}
                              for i,name in enumerate(names)])
        return f,mapping


def extract_35_wav_features(archive,mapping,max_seconds=5.0):
    with ZipFile(archive) as z:
        rows=[]
        for row in mapping.itertuples(index=False):
            name=WAV_BASE+row.stimulus_filename
            if name not in z.namelist():raise FileNotFoundError(name)
            wav=z.read(name)
            ft,meta=features_from_wav_bytes(wav,max_seconds=max_seconds)
            rows.append({'stimulus':row.stimulus,
                         **dict(zip(FEATURE_NAMES,ft)),
                         **meta,'wav_sha256':hashlib.sha256(wav).hexdigest()})
        return pd.DataFrame(rows)


def published_design_table():
    """Only settings explicitly listed in Table 4 of the 2025 JAES study.
    No claims for synthesis candidates 19--26: design metadata missing.
    """
    design=[
        ('49/50','200/300','-',1),('50','400/680/1080','-',1),
        ('50','400/680/1080','7.5/15 kHz',1),('50','400/680/1080','-',2),
        ('50','200/340/540','-',1),('50','200/340/540','-',2),
        ('50','Turbine Base','-',1),('49/50','Turbine Base','-',1),
        ('49/50','Turbine Base','-',2),('49/50','400/680/1080','-',1),
        ('49/50','400/680/1080','-',2),('120/50','Turbine Base','-',1),
        ('120/50','400/680/1080','-',1),('120/50','400/680/1080','-',5),
        ('120/50','400/680/1080','-',10),('120/50','Turbine Base','160/960/970',1),
        ('120/50','Turbine Base','160/960/970',5),('50/120','-','Cmaj7',5),
    ]
    return pd.DataFrame([dict(stimulus=f'S{i+1:02d}',sawtooth_hz=s,narrowband_noise=n,sine=sine,
                              lowpass_khz=lp,source='JAES 2025 Table 4; published 18 candidates')
                         for i,(s,n,sine,lp) in enumerate(design)])
