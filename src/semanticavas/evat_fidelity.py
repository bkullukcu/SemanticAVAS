"""Independent EVAT audio fidelity audit (Müller & Kropp, Zenodo 10610491).

Comparisons are *digital audio signal* comparisons. No acoustic pressure
calibration or perceptual-label prediction is inferred from example WAVs.
Recorded and synthesized channels must belong to the same source category.
"""
from __future__ import annotations
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
from collections import defaultdict
import hashlib
import re
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.spatial.distance import jensenshannon

EVAT_ARCHIVE_MD5 = '4734638643512a3a100f7ea6dee158fd'
EVAT_DOI = '10.5281/zenodo.10610491'
_CATEGORIES = ('avas', 'passby', 'tire')
_STATES = {'measured', 'synthesized', 'generated'}


def archive_checksum(path, expected=EVAT_ARCHIVE_MD5, chunk_size=8 * 1024 * 1024):
    md5 = hashlib.md5()  # published file checksum, not a security primitive
    with open(path, 'rb') as fh:
        while b := fh.read(chunk_size):
            md5.update(b)
    return {'computed_md5': md5.hexdigest(), 'expected_md5': expected,
            'matches_published_md5': md5.hexdigest() == expected}


def evat_manifest(path, *, require_all_categories=True):
    """Match only unambiguous measured/synthesized or measured/generated pairs."""
    grouped = defaultdict(dict)
    with ZipFile(path) as z:
        wavs = [v for v in z.infolist() if not v.is_dir() and v.filename.lower().endswith('.wav')]
        for item in wavs:
            parts = Path(item.filename).parts
            if len(parts) != 2 or parts[0] not in _CATEGORIES:
                raise ValueError(f'Unexpected EVAT WAV path: {item.filename}')
            stem = Path(item.filename).stem
            m = re.fullmatch(r'(.+)_(measured|synthesized|generated)', stem)
            if not m:
                raise ValueError(f'Missing or unsupported signal state: {stem}')
            key, state = m.groups()
            if state in grouped[(parts[0], key)]:
                raise ValueError(f'Duplicate state in {key}')
            grouped[(parts[0], key)][state] = item.filename
    if not grouped:
        raise ValueError('No EVAT comparison pairs found')
    rows = []
    for (category, key), states in sorted(grouped.items()):
        if 'measured' not in states or len(states) != 2:
            raise ValueError(f'Unpaired or ambiguous recording: {category}/{key}')
        model_states = sorted(set(states) - {'measured'})
        if len(model_states) != 1:
            raise ValueError(f'Expected exactly one synthesis for {key}')
        state = model_states[0]
        if (category == 'passby' and state != 'generated') or (category != 'passby' and state != 'synthesized'):
            raise ValueError(f'Unexpected synthesis type for {category}/{key}: {state}')
        rows.append({'category':category,'pair_id':key,'measured':states['measured'],
                     'comparison':states[state],'comparison_type':state})
    if require_all_categories and set(_CATEGORIES) != set(x['category'] for x in rows):
        raise ValueError('Missing expected AVAS, passby or tire category')
    return pd.DataFrame(rows)


def _load_audio(raw):
    try:
        import soundfile as sf
    except ImportError as e:
        raise ImportError('EVAT analysis requires the optional dependency soundfile; pip install semanticavas[evat]') from e
    x, fs = sf.read(BytesIO(raw), dtype='float64', always_2d=True)
    if fs < 16000 or x.shape[0] < fs // 2 or x.shape[1] not in (1,2):
        raise ValueError('Invalid audio sample rate, channel count, or duration')
    if not np.isfinite(x).all() or np.max(np.abs(x)) <= 0:
        raise ValueError('Audio includes NaN, Inf, or silence')
    return x, fs


def _powers(x, fs):
    # Channel power averaged, not stereo-to-mono waveform mixing.
    f, p = welch(x, fs=fs, nperseg=8192, noverlap=4096, axis=0, detrend='constant')
    return f, np.mean(p, axis=1)


def _barkless_third_octave(f, p, *, low=100, high=8000):
    # ANSI-like nominal geometric centers; descriptive PSD features only.
    c = 100.0 * 2 ** (np.arange(0, 20) / 3)
    c = c[c <= high * 1.02]  # include nominal 8 kHz center
    energy = []
    for center in c:
        sel = (f >= max(low, center / 2**(1/6))) & (f < min(high, center * 2**(1/6)))
        if sel.sum() < 2:
            raise ValueError('Insufficient FFT frequency resolution for bands')
        energy.append(float(np.sum(p[sel]) * np.median(np.diff(f))))
    return c, np.asarray(energy)


def _rms_db(x):
    return float(20*np.log10(max(np.sqrt(np.mean(x*x)), 1e-14)))


def _envelope(x, fs, step_s=.10):
    # Mean of per-channel energy, avoiding phase cancellation.
    hop=max(1,round(step_s*fs))
    n=len(x)//hop
    if n < 15: return np.array([])
    blocks=x[:n*hop].reshape(n,hop,x.shape[1]); rms=np.sqrt(np.mean(blocks*blocks,axis=(1,2)))
    return 20*np.log10(np.maximum(rms,1e-12))


def best_envelope_correlation(a, b, *, max_shift_frames=30, min_overlap_frames=25):
    """Correlation after bounded shift search; diagnostic, *not* causal matching."""
    a=np.asarray(a,float);b=np.asarray(b,float)
    best=(-np.inf, None, None)
    for lag in range(-max_shift_frames,max_shift_frames+1):
        start_a=max(0,lag);start_b=max(0,-lag)
        n=min(len(a)-start_a,len(b)-start_b)
        if n<min_overlap_frames:continue
        ax=a[start_a:start_a+n]; bx=b[start_b:start_b+n]
        sa=ax.std();sb=bx.std()
        if sa<1e-10 or sb<1e-10:continue
        corr=float(np.corrcoef(ax,bx)[0,1])
        if corr>best[0]:best=(corr,lag,n)
    return (float(best[0]),int(best[1]),int(best[2])) if best[1] is not None else (np.nan,np.nan,0)


def compare_evat_pair(measured_raw, comparison_raw, *,category, pair_id, comparison_type):
    m, sr=_load_audio(measured_raw);s, sr2=_load_audio(comparison_raw)
    if sr!=sr2 or m.shape[1]!=s.shape[1]:
        raise ValueError('Matched files have different rates or channel counts')
    fm,pm=_powers(m,sr);fs,ps=_powers(s,sr)
    if not np.array_equal(fm,fs):raise ValueError('PSD frequency bins mismatch')
    centers, em = _barkless_third_octave(fm,pm)
    _, es = _barkless_third_octave(fs,ps)
    # Shape divergence on finite energy distribution, avoids arbitrary dB floor.
    em=np.maximum(em,0); es=np.maximum(es,0)
    if em.sum()<=0 or es.sum()<=0:raise ValueError('Zero analysis-band energy')
    jsd=float(jensenshannon(em/em.sum(),es/es.sum(),base=2))
    m_rms=_rms_db(m);s_rms=_rms_db(s)
    result={'category':category,'pair_id':pair_id,'comparison_type':comparison_type,
            'sample_rate_hz':sr,'channels':m.shape[1],
            'measured_duration_s':len(m)/sr,'comparison_duration_s':len(s)/sr,
            'measured_rms_db_digital':m_rms,'comparison_rms_db_digital':s_rms,
            'comparison_minus_measured_rms_db':s_rms-m_rms,
            'third_octave_shape_js_distance_100_8000':jsd,
            'measured_centroid_100_8000_hz':float(np.dot(centers,em)/em.sum()),
            'comparison_centroid_100_8000_hz':float(np.dot(centers,es)/es.sum()),
            'envelope_best_pearson_r':np.nan,'envelope_shift_s':np.nan,
            'interaural_level_diff_change_db':np.nan}
    if m.shape[1]==2:
        ild_m=_rms_db(m[:,0])-_rms_db(m[:,1]);ild_s=_rms_db(s[:,0])-_rms_db(s[:,1])
        result['interaural_level_diff_change_db']=ild_s-ild_m
    if category=='passby':
        r,lag,overlap=best_envelope_correlation(_envelope(m,sr),_envelope(s,sr))
        result['envelope_best_pearson_r']=r
        result['envelope_shift_s']=lag*.1 if np.isfinite(lag) else np.nan
        result['envelope_overlap_frames']=overlap
    return result, pd.DataFrame({'category':category,'pair_id':pair_id,
                                'center_hz':centers,'measured_energy':em,
                                'comparison_energy':es,
                                'measured_energy_fraction':em/em.sum(),
                                'comparison_energy_fraction':es/es.sum()})


def benchmark_evat_archive(path, *, require_all_categories=True):
    manifest=evat_manifest(path,require_all_categories=require_all_categories)
    data=[];bands=[]
    with ZipFile(path) as z:
        for row in manifest.to_dict('records'):
            metric,band=compare_evat_pair(z.read(row['measured']),z.read(row['comparison']),
                   category=row['category'],pair_id=row['pair_id'],comparison_type=row['comparison_type'])
            data.append(metric);bands.append(band)
    return manifest,pd.DataFrame(data),pd.concat(bands,ignore_index=True)
