"""Optional E3 psychoacoustics and E4 frozen audio embeddings.

The released WAV predictor remains unchanged. E3 requires verified Pa
calibration; E4 consumes local frozen-model embeddings with SHA-256 checks.
"""
from __future__ import annotations
from hashlib import sha256
from io import BytesIO
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample_poly

E3_NAMES = ('loudness_sone', 'sharpness_acum', 'roughness_asper', 'tnr_db')


def calibrated_mosqito_features(wav_path, *, pa_per_fullscale, metrics=None, max_seconds=5.):
    """Return averaged physical-input metrics for verified calibrated WAVs."""
    gain = np.asarray(pa_per_fullscale, float)
    if gain.ndim != 0 or not np.isfinite(gain) or gain <= 0:
        raise ValueError('E3 requires verified, positive pa_per_fullscale')
    if max_seconds is not None and (not np.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError('max_seconds must be positive or None')
    if metrics is None:
        try:
            from mosqito.sq_metrics import (
                loudness_zwst, sharpness_din_st, roughness_dw, tnr_ecma_st)
        except ImportError as exc:
            raise ImportError('Install optional mosqito for E3') from exc
        metrics = (loudness_zwst, sharpness_din_st, roughness_dw, tnr_ecma_st)
    path = Path(wav_path)
    if not path.is_file() or path.suffix.lower() != '.wav' or path.stat().st_size > 100_000_000:
        raise ValueError('Existing WAV smaller than 100 MB required')
    fs, raw = wavfile.read(BytesIO(path.read_bytes()))
    if fs < 8000 or raw.ndim not in (1,2):
        raise ValueError('Unsupported WAV layout or sample rate')
    signal = raw.astype(np.float64)
    if raw.dtype.kind == 'u':
        mid=(np.iinfo(raw.dtype).max+1)/2
        signal=(signal-mid)/mid
    elif raw.dtype.kind == 'i':
        lim=max(abs(np.iinfo(raw.dtype).min),np.iinfo(raw.dtype).max)
        signal=signal/lim
    if max_seconds is not None:
        signal=signal[:int(fs*max_seconds)]
    if len(signal)<4096 or not np.isfinite(signal).all():
        raise ValueError('Invalid recording')
    if signal.ndim==1:
        signal=signal[:,None]
    outputs=[]
    for channel in signal.T:
        pa=channel*float(gain)
        if fs!=48000:
            d=gcd(fs,48000)
            pa=resample_poly(pa,48000//d,fs//d)
        if float(np.sqrt(np.mean(pa**2)))<1e-12:
            raise ValueError('Silence cannot be analyzed')
        n=metrics[0](pa,48000)[0]
        s=metrics[1](pa,48000)
        r=metrics[2](pa,48000)[0]
        t=metrics[3](pa,48000)[0]
        row=[float(np.mean(np.asarray(value,float))) for value in (n,s,r,t)]
        if not np.isfinite(row).all():
            raise ValueError('Nonfinite psychoacoustic result')
        outputs.append(row)
    return dict(zip(E3_NAMES,np.mean(outputs,axis=0).astype(float).tolist()))


def fixed_clap_embeddings(paths, stimulus_ids, *, checkpoint_path, model=None):
    """Extract embeddings with locally supplied CLAP weights; never auto-download."""
    paths=[Path(p) for p in paths]
    ids=list(map(str,stimulus_ids))
    checkpoint=Path(checkpoint_path)
    if not checkpoint.is_file() or len(paths)!=len(ids) or len(set(ids))!=len(ids) or not ids:
        raise ValueError('Local checkpoint and distinct aligned stimulus IDs required')
    if any(not p.is_file() or p.suffix.lower()!='.wav' for p in paths):
        raise ValueError('All stimulus WAVs must exist')
    if model is None:
        try:
            import laion_clap
        except ImportError as exc:
            raise ImportError('Install laion-clap in an isolated environment') from exc
        model=laion_clap.CLAP_Module(enable_fusion=False)
        model.load_ckpt(str(checkpoint))
    matrix=np.asarray(model.get_audio_embedding_from_filelist(
        x=[str(p) for p in paths],use_tensor=False),float)
    if matrix.ndim!=2 or len(matrix)!=len(paths) or matrix.shape[1]<8 or not np.isfinite(matrix).all():
        raise ValueError('Invalid CLAP embedding output')
    table=pd.DataFrame(matrix,columns=[f'clap_{i:04d}' for i in range(matrix.shape[1])])
    table.insert(0,'wav_sha256',[sha256(p.read_bytes()).hexdigest() for p in paths])
    table.insert(0,'stimulus',ids)
    return table, {'checkpoint_sha256':sha256(checkpoint.read_bytes()).hexdigest(),
                   'dimension':int(matrix.shape[1]),'n_sounds':len(ids)}


def checked_embedding_matrix(frame, stimuli, wav_hashes):
    """Exact stimulus and WAV provenance matching, independent of row order."""
    columns=sorted([c for c in frame.columns if c.startswith('clap_')])
    stimuli=list(map(str,stimuli))
    if len(columns)<8 or not {'stimulus','wav_sha256'}.issubset(frame.columns):
        raise ValueError('Embedding schema needs stimulus, wav_sha256, clap_0000...')
    if frame.stimulus.duplicated().any() or set(frame.stimulus)!=set(stimuli):
        raise ValueError('Stimulus keys must match exactly')
    ordered=frame.set_index('stimulus').loc[stimuli]
    if ordered.wav_sha256.tolist()!=list(wav_hashes):
        raise ValueError('CLAP and WAV checksum mismatch')
    x=ordered[columns].to_numpy(float)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite CLAP data')
    return x


def nested_representation_test(representations, y, groups, *, alphas=(.1,1.,10.,100.),
                               pca_components=(2,4), low=0., high=100.):
    """Train-only scaling/PCA; nested leave-one-source-group-out evaluation."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    y=np.asarray(y,float); groups=np.asarray(groups,str)
    if y.ndim!=2 or len(groups)!=len(y) or len(np.unique(groups))<3 or not np.isfinite(y).all():
        raise ValueError('At least three independent source families required')
    lower=np.broadcast_to(np.asarray(low,float),(y.shape[1],))
    upper=np.broadcast_to(np.asarray(high,float),(y.shape[1],))
    if np.any(lower>=upper) or not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError('Invalid target scale bounds')
    def estimator(alpha,k):
        steps=[StandardScaler()]
        if k is not None:steps.append(PCA(n_components=k))
        return make_pipeline(*steps,Ridge(alpha=float(alpha)))
    records=[]
    indices=np.arange(len(y))
    for label,values in representations.items():
        x=np.asarray(values,float)
        if x.ndim!=2 or len(x)!=len(y) or not np.isfinite(x).all():
            raise ValueError('Unaligned representation')
        for held in np.unique(groups):
            test=indices[groups==held];train=indices[groups!=held]
            choices=[(a,k) for a in alphas for k in (None,*pca_components)]
            scores=[]
            for alpha,k in choices:
                errors=[]
                for val_grp in np.unique(groups[train]):
                    valid=train[groups[train]==val_grp]
                    fit=train[groups[train]!=val_grp]
                    if len(fit)<3 or (k is not None and k>min(len(fit),x.shape[1])):
                        errors=[np.inf];break
                    model=estimator(alpha,k).fit(x[fit],y[fit])
                    pred=np.clip(model.predict(x[valid]),lower,upper)
                    errors.append(float(np.abs(pred-y[valid]).mean()))
                scores.append(float(np.mean(errors)))
            if not np.isfinite(scores).any():
                raise ValueError('Not enough training groups for nested fitting')
            alpha,k=choices[int(np.argmin(scores))]
            model=estimator(alpha,k).fit(x[train],y[train])
            pred=np.clip(model.predict(x[test]),lower,upper)
            base=y[train].mean(axis=0)
            for index,value in zip(test,pred):
                records.append({'representation':label,'source_group':str(held),
                    'sample_index':int(index),'alpha':float(alpha),
                    'pca_components':-1 if k is None else int(k),
                    'mae':float(np.abs(value-y[index]).mean()),
                    'baseline_mae':float(np.abs(base-y[index]).mean())})
    predictions=pd.DataFrame(records)
    summary=predictions.groupby('representation')[['mae','baseline_mae']].mean().reset_index()
    summary['delta_mae']=summary.mae-summary.baseline_mae
    return predictions,summary
