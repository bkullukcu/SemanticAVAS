"""Uncalibrated digital-audio descriptors; NOT loudness, sharpness, or SPL."""
from __future__ import annotations
import numpy as np
from scipy.signal import welch
from scipy.io import wavfile
from io import BytesIO

def features_from_wav_bytes(data, *, max_seconds=5.0):
    fs,wav=wavfile.read(BytesIO(data))
    if wav.ndim not in (1,2) or fs<8000:raise ValueError('unsupported digital PCM shape/sampling rate')
    x=wav.astype(np.float64)
    if wav.dtype.kind == 'u':
        # 8-bit PCM is unsigned; silence is represented by the midpoint.
        midpoint=(np.iinfo(wav.dtype).max+1)/2
        x=(x-midpoint)/midpoint
    elif wav.dtype.kind == 'i':
        scale=max(abs(np.iinfo(wav.dtype).min),np.iinfo(wav.dtype).max)
        x/=scale
    x=x[:min(len(x),int(fs*max_seconds))]
    if x.ndim==1:x=x[:,None]
    if not np.isfinite(x).all() or x.shape[0]<4096:raise ValueError('invalid audio')
    # Per-channel descriptors are averaged; DO NOT phase-cancel stereo channels.
    chans=[]
    for i in range(x.shape[1]):
        sig=x[:,i]
        rms=np.sqrt(np.mean(sig**2))
        if rms<=1e-12:continue
        freq,psd=welch(sig,fs=fs,nperseg=min(4096,len(sig)),noverlap=2048)
        psd=np.maximum(psd,0)
        valid=(freq>=20)&(freq<=min(10000,fs/2))
        f=freq[valid];p=psd[valid]
        tot=p.sum()
        if tot<=0:continue
        v=p/tot
        cent=float(np.dot(f,v));spread=float(np.sqrt(np.dot((f-cent)**2,v)))
        cdf=np.cumsum(v);roll=float(f[np.searchsorted(cdf,.9)])
        low=float(v[f<500].sum());mid=float(v[(f>=500)&(f<2000)].sum());high=float(v[f>=2000].sum())
        flat=float(np.exp(np.log(np.maximum(p,1e-20)).mean())/(p.mean()+1e-20))
        crest=float(np.max(abs(sig))/rms)
        zcr=float(np.mean((sig[:-1]<0)!=(sig[1:]<0)))
        chans.append([20*np.log10(rms),cent,spread,roll,low,mid,high,flat,crest,zcr])
    if not chans:raise ValueError('no non-silent channels')
    return np.mean(chans,axis=0).tolist(),{'sample_rate_hz':int(fs),'duration_s':round(len(x)/fs,5),'channels':int(x.shape[1])}

FEATURE_NAMES=['rms_dbfs','spectral_centroid_hz','spectral_spread_hz','rolloff_90_hz',
               'energy_lt_500_fraction','energy_500_2000_fraction','energy_ge_2000_fraction',
               'spectral_flatness','crest_factor','zero_crossing_rate']
