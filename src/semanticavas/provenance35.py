"""Conservative identity and provenance checks for the JAES 35-sound experiment.

Neither HDF importer information nor WAV spectral descriptors prove the
complete synthesis settings of a stimulus.
"""
from __future__ import annotations
from pathlib import Path
from zipfile import ZipFile
import hashlib
import re
import numpy as np
import pandas as pd

AUDIO_ROOT = 'Sounds_Pedestrian_Listening_Test/'

def audio_inventory(archive, mapping: pd.DataFrame) -> pd.DataFrame:
    """Audit canonical sound IDs, WAV bytes and optional HEAD HDF importer labels.

    Mapping must derive from both MAT session sound lists, not guessed ordering.
    Does not publish or extract participant-level data.
    """
    required={'stimulus','stimulus_filename'}
    if not required.issubset(mapping.columns):
        raise ValueError('expected stimulus and stimulus_filename')
    if len(mapping)!=35 or mapping['stimulus'].nunique()!=35 or mapping['stimulus_filename'].nunique()!=35:
        raise ValueError('expected 35 unique indexed assets')
    result=[]
    with ZipFile(archive) as z:
        names=set(z.namelist())
        for row in mapping.itertuples(index=False):
            wav_name=AUDIO_ROOT+row.stimulus_filename
            if wav_name not in names: raise FileNotFoundError(wav_name)
            wave=z.read(wav_name)
            stem=Path(row.stimulus_filename).stem
            is_numbered=bool(re.fullmatch(r'\d+',stem))
            no=int(row.stimulus[1:])
            if is_numbered and (no>26 or stem!=str(no)):
                raise ValueError(f'misaligned numbered WAV {row.stimulus} -> {row.stimulus_filename}')
            if not is_numbered and no<=26:
                raise ValueError(f'misaligned named WAV {row.stimulus}')
            hdf_name=AUDIO_ROOT+stem+'.hdf' if is_numbered else None
            hdf_imported_from_wav=False
            hdf_channel_unit=None
            if hdf_name in names:
                with z.open(hdf_name) as file:
                    header=file.read(65536).decode('cp1252','replace')
                hdf_imported_from_wav=bool(re.search(r"Imported from '\d+\.wav'",header))
                units=re.findall(r'^physical unit:\s*(\S+)',header,re.MULTILINE)
                hdf_channel_unit=units[1] if len(units)>1 else None
            # Only an explicit synthesis graph / recipe counts as provenance.
            result.append(dict(stimulus=row.stimulus,stimulus_filename=row.stimulus_filename,
                           category_from_filename='numbered_asset' if is_numbered else 'named_asset',
                           wav_sha256=hashlib.sha256(wave).hexdigest(),
                           file_has_hdf=hdf_name in names,
                           hdf_imported_from_wav=hdf_imported_from_wav,
                           hdf_channel_unit=hdf_channel_unit,
                           full_recipe_verified=False,
                           published_partial_recipe=(no<=18 and is_numbered),
                           external_playback_spl_verified=False,
                           synthesis_source_type_confirmed=False))
    r=pd.DataFrame(result)
    if int((r.category_from_filename=='numbered_asset').sum())!=26 or int((r.category_from_filename=='named_asset').sum())!=9:
        raise ValueError('archive must contain 26 numbered and nine named sources')
    if r.wav_sha256.nunique()!=35:
        raise ValueError('duplicate WAV bytes found; inspect before analysis')
    return r

def nearest_published_descriptor_neighbors(feature_df, feature_cols):
    """Find approximate neighbors in standardized acoustic descriptor space.

    This is *not* synthesis parameter inversion or proof of related source.
    """
    x=feature_df.set_index('stimulus').sort_index()
    required=set(feature_cols)
    if not required.issubset(x.columns): raise ValueError('missing features')
    vals=x.loc[[f'S{i:02d}' for i in range(1,27)],list(feature_cols)].astype(float)
    if not np.isfinite(vals.to_numpy()).all(): raise ValueError('nonfinite audio features')
    mu=vals.mean(axis=0)
    sd=vals.std(axis=0,ddof=0).replace(0,1)
    z=(vals-mu)/sd
    known=z.loc[[f'S{i:02d}' for i in range(1,19)]].to_numpy()
    records=[]
    for i in range(19,27):
        sid=f'S{i:02d}'
        dist=np.linalg.norm(known-z.loc[sid].to_numpy(),axis=1)
        ix=int(np.argmin(dist))
        records.append({'stimulus':sid,'closest_published_descriptor_neighbor':f'S{ix+1:02d}',
                        'standardized_descriptor_distance':float(dist[ix]),
                        'interpretation':'descriptor similarity only; NOT a synthesis recipe'})
    return pd.DataFrame(records)

def group_cv_summary(predictions:pd.DataFrame, scales, *, kind='group'):
    """Per-source-block mean absolute error and reference delta."""
    rows=[]
    for name,g in predictions.groupby('group',sort=True):
        m=np.column_stack([np.abs(g[f'{s}_measured'].values-g[f'{s}_{kind}_prediction'].values) for s in scales])
        b=np.column_stack([np.abs(g[f'{s}_measured'].values-g[f'{s}_{kind}_baseline'].values) for s in scales])
        rows.append(dict(group=name,n_sounds=len(g),model_mae=float(m.mean()),baseline_mae=float(b.mean()),
                         model_minus_baseline=float(m.mean()-b.mean())))
    return pd.DataFrame(rows)
