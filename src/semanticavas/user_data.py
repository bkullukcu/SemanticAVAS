"""Convert third-party semantic ratings into the stable SemanticAtlas table.

No automatic polarity changes or cross-study harmonisation. Numeric scale
bounds and response handling are explicit in the user-owned JSON contract.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd


class InputContractError(ValueError):
    pass


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _config(path):
    with open(path, encoding='utf-8') as f:
        cfg = json.load(f)
    specs = cfg.get('scales')
    if not isinstance(specs, dict) or not specs:
        raise InputContractError('scales is a nonempty mapping of scale names to bounds/direction')
    for scale, spec in specs.items():
        if not isinstance(spec, dict) or not {'min', 'max', 'direction'} <= set(spec):
            raise InputContractError(f'scale {scale} requires min/max/direction')
        lo, hi = float(spec['min']), float(spec['max'])
        if not np.isfinite([lo, hi]).all() or lo >= hi or spec['direction'] not in ('higher_is_more', 'higher_is_less'):
            raise InputContractError(f'invalid bounds or polarity for {scale}')
    if cfg.get('missing_policy', 'error') not in ('error', 'drop_incomplete'):
        raise InputContractError('missing_policy must be error or drop_incomplete')
    return cfg


def _ids(values, name):
    s = pd.Series(values)
    if s.isna().any() or s.astype(str).str.strip().eq('').any():
        raise InputContractError(f'null/empty {name}')
    return s.astype(str).str.strip().to_numpy()


def _long_to_wide(df, cfg):
    mapping = cfg.get('columns', {})
    mode = cfg.get('layout', 'wide')
    if mode not in ('wide', 'long'):
        raise InputContractError('layout must be wide or long')
    id_names = ('participant', 'stimulus')
    cols = [mapping.get(k, k) for k in id_names]
    if not set(cols) <= set(df):
        raise InputContractError('participant and stimulus columns required')
    participant = _ids(df[cols[0]], 'participant')
    stimulus = _ids(df[cols[1]], 'stimulus')
    src = pd.DataFrame({'participant': participant, 'stimulus': stimulus})
    trial_col = mapping.get('trial', 'trial')
    if trial_col in df:
        src['trial'] = _ids(df[trial_col], 'trial')
    elif cfg.get('allow_repeated_ratings', False):
        raise InputContractError('allow_repeated_ratings requires explicit trial column')
    scales = list(cfg['scales'])
    if mode == 'wide':
        for k in scales:
            c = mapping.get(k, k)
            if c not in df:
                raise InputContractError(f'missing rating column {c}')
            src[k] = pd.to_numeric(df[c], errors='coerce').to_numpy()
    else:
        rating_col = mapping.get('rating', 'rating')
        scale_col = mapping.get('scale', 'scale')
        if rating_col not in df or scale_col not in df:
            raise InputContractError('long form requires scale and rating columns')
        if df[scale_col].isna().any() or not set(df[scale_col].astype(str)) <= set(scales):
            raise InputContractError('unrecognised/null rating scale')
        src['scale'] = df[scale_col].astype(str).to_numpy()
        src['rating'] = pd.to_numeric(df[rating_col], errors='coerce').to_numpy()
        keys = ['participant', 'stimulus'] + (['trial'] if 'trial' in src else [])
        if src.duplicated(keys + ['scale']).any():
            raise InputContractError('duplicate subject/stimulus/trial/scale ratings')
        src = src.pivot(index=keys, columns='scale', values='rating').reset_index()
        src.columns.name = None
        for k in scales:
            if k not in src: src[k] = np.nan
    keys = ['participant', 'stimulus'] + (['trial'] if 'trial' in src else [])
    if src.duplicated(keys).any():
        raise InputContractError('duplicate participant/stimulus/trial; supply unique trial IDs')
    if not cfg.get('allow_repeated_ratings', False) and src.duplicated(['participant', 'stimulus']).any():
        raise InputContractError('repeat ratings found; opt in and supply trial IDs')
    return src[keys + scales]


def _finish(df, input_path, schema_path, output_dir, cfg, source_type):
    scales = list(cfg['scales'])
    n_before = len(df)
    for s in scales:
        df[s] = pd.to_numeric(df[s], errors='coerce')
        values = df[s].to_numpy(float)
        spec = cfg['scales'][s]
        good = ~np.isnan(values)
        if not np.isfinite(values[good]).all() or np.any(values[good] < float(spec['min'])) or np.any(values[good] > float(spec['max'])):
            raise InputContractError(f'{s}: ratings must be finite and inside declared range')
    if cfg.get('missing_policy', 'error') == 'drop_incomplete':
        df = df.dropna(subset=scales).copy()
    elif df[scales].isna().any().any():
        raise InputContractError('missing rating; configure missing_policy=drop_incomplete explicitly')
    if df.empty or df['participant'].nunique() < 2 or df['stimulus'].nunique() < 2:
        raise InputContractError('at least two participants and two stimuli with complete ratings required')
    df = df.sort_values(['participant', 'stimulus'] + (['trial'] if 'trial' in df else []), kind='stable').reset_index(drop=True)
    from .core import SemanticAtlas
    # End-to-end compatibility check with the existing ranking model.
    atlas = SemanticAtlas(df, scales)
    if len(atlas.means) < 2:
        raise InputContractError('insufficient unique stimuli')
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    ratings_path = out / 'ratings.csv'
    if ratings_path.resolve() == Path(input_path).resolve():
        raise InputContractError('cannot overwrite raw source data')
    # Optional metadata must be a separate, explicit stimulus key, not guessed from file order.
    metadata_file = cfg.get('metadata_csv')
    metadata_details = {'metadata_included': False}
    if metadata_file:
        path = Path(metadata_file)
        if not path.is_absolute(): path = Path(schema_path).parent / path
        metadata = pd.read_csv(path)
        id_col = cfg.get('metadata_stimulus_column', 'stimulus')
        if id_col not in metadata.columns:
            raise InputContractError('metadata stimulus ID column absent')
        metadata = metadata.rename(columns={id_col: 'stimulus'})
        metadata['stimulus'] = _ids(metadata['stimulus'], 'metadata stimulus')
        if metadata['stimulus'].duplicated().any() or set(df['stimulus']) - set(metadata['stimulus']):
            raise InputContractError('metadata IDs must be unique and cover every rated stimulus')
        metadata = metadata[metadata['stimulus'].isin(df['stimulus'])].sort_values('stimulus')
        metadata.to_csv(out / 'stimulus_metadata.csv', index=False)
        metadata_details = {'metadata_included': True, 'metadata_sha256': _sha(path), 'metadata_file': path.name}
    df.to_csv(ratings_path, index=False)
    coverage = df[['participant','stimulus']].drop_duplicates().groupby('stimulus').size().rename('participants_with_rating').reset_index()
    coverage.to_csv(out / 'coverage_by_stimulus.csv', index=False)
    full_grid = int(df['participant'].nunique() * df['stimulus'].nunique())
    observed_pairs = len(df[['participant','stimulus']].drop_duplicates())
    manifest = {
        'schema_version': 'semanticavas-user-data-v1',
        'input_type': source_type,
        'source_file': Path(input_path).name, 'source_sha256': _sha(input_path),
        'mapping_file': Path(schema_path).name, 'mapping_sha256': _sha(schema_path),
        'output_file': ratings_path.name, 'output_sha256': _sha(ratings_path),
        'rows_before_policy': n_before, 'rows_after_policy': len(df),
        'rows_dropped': n_before - len(df),
        'distinct_participant_stimulus_pairs': observed_pairs,
        'missing_participant_stimulus_pairs': full_grid - observed_pairs,
        'complete_repeated_measures_design': observed_pairs == full_grid,
        'participants': int(df['participant'].nunique()), 'stimuli': int(df['stimulus'].nunique()),
        'scales': cfg['scales'], 'rating_polarity_was_flipped': False,
        'cross_study_scale_comparability': 'not established',
        'audio_calibration': 'not inferred from ratings',
        'repetitions_preserved': 'trial' in df,
        **metadata_details,
    }
    (out / 'conversion_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return manifest


def convert_csv(input_path, schema_path, output_dir):
    cfg = _config(schema_path)
    df = pd.read_csv(input_path)
    return _finish(_long_to_wide(df, cfg), input_path, schema_path, output_dir, cfg, 'csv')


def convert_npz(input_path, schema_path, output_dir):
    cfg = _config(schema_path)
    with np.load(input_path, allow_pickle=False) as z:
        required = {'ratings', 'participants', 'stimuli', 'scales'}
        if not required <= set(z.files):
            raise InputContractError(f'NPZ missing {sorted(required - set(z.files))}')
        values = np.array(z['ratings'])
        try:
            participants = np.array(z['participants'])
            stimuli = np.array(z['stimuli'])
            scales = np.array(z['scales']).astype(str).tolist()
            trials = np.array(z['trials']) if 'trials' in z else None
        except ValueError as e:
            raise InputContractError('NPZ must use Unicode string arrays (dtype=str), not object/pickle arrays') from e
    if values.ndim != 2 or values.shape[0] != len(participants) or values.shape[0] != len(stimuli) or values.shape[1] != len(scales) or set(scales) != set(cfg['scales']) or len(set(scales)) != len(scales):
        raise InputContractError('ratings shape must be (observation, explicit scale); names must match the schema')
    if trials is not None and (len(trials) != len(values) or not cfg.get('allow_repeated_ratings', False)):
        raise InputContractError('trial IDs require allow_repeated_ratings=true')
    df = pd.DataFrame(values, columns=scales)
    df.insert(0, 'stimulus', _ids(stimuli, 'stimulus'))
    df.insert(0, 'participant', _ids(participants, 'participant'))
    if trials is not None: df['trial'] = _ids(trials, 'trial')
    if df.duplicated(['participant', 'stimulus'] + (['trial'] if trials is not None else [])).any():
        raise InputContractError('duplicate participant/stimulus/trial')
    if trials is None and df.duplicated(['participant', 'stimulus']).any():
        raise InputContractError('repeated observations require explicit trial IDs')
    return _finish(df, input_path, schema_path, output_dir, cfg, 'npz')


def convert(input_path, schema_path, output_dir):
    suffix = Path(input_path).suffix.lower()
    if suffix == '.csv': return convert_csv(input_path, schema_path, output_dir)
    if suffix == '.npz': return convert_npz(input_path, schema_path, output_dir)
    raise InputContractError('supported inputs: .csv and .npz')
