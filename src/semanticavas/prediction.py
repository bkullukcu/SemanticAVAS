"""Train and apply an acoustic-to-semantic ridge predictor for WAV files.

The unit of generalisation is a sound (or a declared recording family), rather
than an individual listener response. All folds fit their own standardisation.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .acoustic_features import FEATURE_NAMES, features_from_wav_bytes
from .empirical import nested_leave_one_group_out, nested_leave_one_sound_out, _ridge_fit_predict

ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
FORMAT_VERSION = 1


def _matrix(values, name):
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError(f'{name} requires a finite two-dimensional matrix')
    return result


def _read_wav(path: Path):
    if path.suffix.lower() != '.wav':
        raise ValueError(f'WAV input required: {path}')
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size > 100_000_000:
        raise ValueError(f'Audio file exceeds 100 MB: {path}')
    blob = path.read_bytes()
    return blob, sha256(blob).hexdigest()


def extract_wav(path: str | Path, *, max_seconds: float = 5.0):
    """Return aligned acoustic descriptors and recording metadata for one WAV."""
    if not (np.isfinite(max_seconds) and max_seconds > 0):
        raise ValueError('max_seconds must be positive and finite')
    blob, digest = _read_wav(Path(path))
    vector, meta = features_from_wav_bytes(blob, max_seconds=max_seconds)
    x = np.asarray(vector, dtype=float)
    if x.shape != (len(FEATURE_NAMES),) or not np.isfinite(x).all():
        raise ValueError('Feature extraction returned an unexpected shape')
    return x, {**meta, 'sha256': digest}


def prepare_training(manifest_path: str | Path, ratings_path: str | Path,
                     scales: Sequence[str], *, max_seconds: float = 5.0):
    """Join one WAV per sound to balanced participant ratings.

    manifest.csv: stimulus,wav[,sound_family]. Relative WAV paths start at the
    manifest directory. ratings.csv: participant,stimulus,<named scales>.
    Repeated trials are first averaged within each participant and sound.
    """
    scales = list(scales)
    if not scales or len(set(scales)) != len(scales) or any(not s or s in ('stimulus', 'participant') for s in scales):
        raise ValueError('Provide unique rating-scale columns')
    manifest_path = Path(manifest_path)
    manifest = pd.read_csv(manifest_path, dtype={'stimulus': str, 'wav': str, 'sound_family': str})
    ratings = pd.read_csv(ratings_path, dtype={'stimulus': str, 'participant': str})
    if not {'stimulus', 'wav'}.issubset(manifest) or not {'stimulus', 'participant', *scales}.issubset(ratings):
        raise ValueError('Manifest or ratings table is missing required columns')
    if len(manifest) < 10 or manifest['stimulus'].isna().any() or manifest['wav'].isna().any():
        raise ValueError('Training requires >=10 uniquely named sounds with WAV paths')
    if manifest['stimulus'].duplicated().any() or (manifest['stimulus'].str.strip() == '').any():
        raise ValueError('Manifest stimulus IDs must be unique and nonempty')
    if ratings[['stimulus', 'participant']].isna().any().any():
        raise ValueError('Ratings require participant and stimulus IDs')
    if any((ratings[c].astype(str).str.strip() == '').any() for c in ('participant', 'stimulus')):
        raise ValueError('Ratings contain empty participant or stimulus IDs')
    if set(ratings['stimulus']) != set(manifest['stimulus']):
        raise ValueError('Exact correspondence between WAV stimuli and rated stimuli is required')
    for scale in scales:
        values = pd.to_numeric(ratings[scale], errors='raise').to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 100)):
            raise ValueError(f'{scale}: use finite participant ratings on the declared 0–100 scale')
        ratings[scale] = values
    grouped = ratings.groupby(['stimulus', 'participant'], sort=True, as_index=False)[scales].mean()
    sizes = grouped.groupby('stimulus')['participant'].nunique()
    if (sizes < 2).any():
        raise ValueError('At least two distinct listeners per sound are required')
    profiles = grouped.groupby('stimulus', sort=True)[scales].mean()
    ordered = manifest.sort_values('stimulus').reset_index(drop=True)
    fam = None
    if 'sound_family' in ordered:
        if ordered['sound_family'].isna().any() or (ordered['sound_family'].str.strip() == '').any():
            raise ValueError('sound_family must be complete when supplied')
        fam = ordered['sound_family'].to_numpy(str)
        if len(set(fam)) < 3:
            raise ValueError('Family-blocked validation requires at least three sound families')
    vectors, hashes, meta_rows = [], set(), []
    for row in ordered.itertuples(index=False):
        path = Path(row.wav)
        if not path.is_absolute():
            path = manifest_path.resolve().parent / path
        vector, meta = extract_wav(path, max_seconds=max_seconds)
        if meta['sha256'] in hashes:
            raise ValueError('The same WAV content is associated with multiple stimuli')
        hashes.add(meta['sha256'])
        vectors.append(vector)
        meta_rows.append({'stimulus': row.stimulus, 'sampling_rate_hz': meta['sample_rate_hz'],
                          'duration_s': meta['duration_s'], 'channels': meta['channels'],
                          'listener_count': int(sizes.loc[row.stimulus])})
    x = _matrix(vectors, 'extracted audio features')
    y = _matrix(profiles.loc[ordered['stimulus'], scales].to_numpy(float), 'balanced ratings')
    return x, y, fam, pd.DataFrame(meta_rows)


def _choose_alpha(x, y, groups, alphas):
    """Tune final-fit alpha using only training sounds, with group-aware folds."""
    n = len(x)
    folds = ([np.flatnonzero(groups == g) for g in np.unique(groups)]
             if groups is not None else [np.asarray([k]) for k in range(n)])
    losses = []
    for alpha in alphas:
        fold_errors = []
        for test in folds:
            train = np.setdiff1d(np.arange(n), test)
            if len(train) < 3:
                raise ValueError('Too few training sounds for inner validation')
            prediction = np.clip(_ridge_fit_predict(x[train], y[train], x[test], alpha), 0, 100)
            fold_errors.append(float(np.abs(prediction - y[test]).mean()))
        losses.append(float(np.mean(fold_errors)))
    return float(alphas[int(np.argmin(losses))])


@dataclass
class WavSemanticPredictor:
    """One fitted multi-output ridge model with training-feature reference data."""
    scales: tuple[str, ...]
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    coefficients: np.ndarray
    reference_z: np.ndarray
    min_features: np.ndarray
    max_features: np.ndarray
    alpha: float
    max_seconds: float
    validation: dict

    def predict_features(self, x):
        x = _matrix(x, 'features')
        if x.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f'Expected {len(FEATURE_NAMES)} acoustic features in FEATURE_NAMES order')
        z = (x - self.x_mean) / self.x_scale
        raw = z @ self.coefficients + self.y_mean
        nearest = np.min(np.sqrt(np.sum((z[:, None, :] - self.reference_z[None, :, :]) ** 2, axis=-1)), axis=1)
        out = []
        for i in range(len(x)):
            item = {name: float(value) for name, value in zip(self.scales, np.clip(raw[i], 0, 100))}
            item.update({
                'outside_training_feature_ranges': bool(np.any((x[i] < self.min_features) | (x[i] > self.max_features))),
                'nearest_training_distance_standardized': float(nearest[i]),
                'prediction_clipped_to_rating_scale': bool(np.any((raw[i] < 0) | (raw[i] > 100))),
            })
            out.append(item)
        return out

    def predict_wav(self, path: str | Path):
        vector, meta = extract_wav(path, max_seconds=self.max_seconds)
        return {'wav': str(path), **self.predict_features(vector[None, :])[0],
                'sample_rate_seen_in_training': int(meta['sample_rate_hz']) in self.validation.get('training_sample_rates_hz', []),
                'audio': meta,
                'cv_abs_error_p90_by_scale': self.validation['cv_abs_error_p90_by_scale']}


def train_wav_predictor(manifest_path, ratings_path, scales, *, max_seconds=5.0, alphas=ALPHAS):
    """Train, perform nested sound/family-held-out evaluation, and return artifacts.

    Output: (fitted_model, per_sound_validation_table, per_sound_audio_summary).
    """
    alphas = tuple(float(a) for a in alphas)
    if not alphas or any(not np.isfinite(a) or a <= 0 for a in alphas):
        raise ValueError('All ridge alphas must be strictly positive')
    x, y, groups, audio_table = prepare_training(manifest_path, ratings_path, scales, max_seconds=max_seconds)
    if groups is None:
        cv_pred, cv_baseline, cv_alpha = nested_leave_one_sound_out(x, y, alpha_grid=alphas)
        cv_method = 'nested_leave_one_sound_out'
    else:
        cv_pred, cv_baseline, cv_alpha = nested_leave_one_group_out(x, y, groups, alpha_grid=alphas)
        cv_method = 'nested_leave_one_family_out'
    cv_pred = np.clip(cv_pred, 0, 100)
    absolute = np.abs(cv_pred - y)
    baseline_absolute = np.abs(cv_baseline - y)
    scale_names = tuple(str(s) for s in scales)
    metrics = {
        'strategy': cv_method,
        'sounds': int(len(x)),
        'families': None if groups is None else int(len(set(groups))),
        'scale_names': list(scale_names),
        'feature_names': list(FEATURE_NAMES),
        'cv_mae_by_scale': dict(zip(scale_names, absolute.mean(axis=0).tolist())),
        'baseline_cv_mae_by_scale': dict(zip(scale_names, baseline_absolute.mean(axis=0).tolist())),
        'cv_abs_error_p90_by_scale': dict(zip(scale_names, np.quantile(absolute, 0.9, axis=0).tolist())),
        'mean_cv_mae': float(absolute.mean()),
        'mean_baseline_cv_mae': float(baseline_absolute.mean()),
        'training_sample_rates_hz': sorted(set(audio_table['sampling_rate_hz'].astype(int).tolist())),
    }
    alpha = _choose_alpha(x, y, groups, alphas)
    x_mean = x.mean(axis=0)
    x_scale = x.std(axis=0)
    x_scale[x_scale < 1e-12] = 1.0
    z = (x - x_mean) / x_scale
    y_mean = y.mean(axis=0)
    coefficients = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - y_mean))
    model = WavSemanticPredictor(scale_names, x_mean, x_scale, y_mean, coefficients,
                                 z, x.min(axis=0), x.max(axis=0), alpha, float(max_seconds), metrics)
    oof = pd.DataFrame({'stimulus': audio_table['stimulus']})
    if groups is not None:
        oof['sound_family'] = groups
    oof['chosen_alpha_outer_fold'] = cv_alpha
    for i, col in enumerate(scale_names):
        oof[f'observed_{col}'] = y[:, i]
        oof[f'predicted_{col}'] = cv_pred[:, i]
        oof[f'baseline_{col}'] = cv_baseline[:, i]
    return model, oof, audio_table


def save_predictor(model: WavSemanticPredictor, path: str | Path):
    """Portable, pickle-free NPZ with an explicit feature schema and diagnostics."""
    metadata = {'format_version': FORMAT_VERSION, 'feature_names': list(FEATURE_NAMES),
                'scales': list(model.scales), 'alpha': model.alpha,
                'max_seconds': model.max_seconds, 'validation': model.validation}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                        x_mean=model.x_mean, x_scale=model.x_scale, y_mean=model.y_mean,
                        coefficients=model.coefficients, reference_z=model.reference_z,
                        min_features=model.min_features, max_features=model.max_features)
    return path


def load_predictor(path: str | Path):
    """Load and validate an NPZ model without executing serialized Python objects."""
    expected = {'metadata', 'x_mean', 'x_scale', 'y_mean', 'coefficients',
                'reference_z', 'min_features', 'max_features'}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected:
            raise ValueError('Incomplete or unrecognized model archive')
        metadata = json.loads(str(archive['metadata'].item()))
        arrays = {key: np.asarray(archive[key], dtype=float).copy() for key in expected - {'metadata'}}
    if metadata.get('format_version') != FORMAT_VERSION or metadata.get('feature_names') != list(FEATURE_NAMES):
        raise ValueError('Incompatible model format or acoustic feature extractor')
    scales = metadata.get('scales')
    n, n_scales = len(FEATURE_NAMES), len(scales) if isinstance(scales, list) else 0
    if (not n_scales or len(set(scales)) != n_scales or
        any(arr.shape != (n,) for arr in [arrays['x_mean'], arrays['x_scale'], arrays['min_features'], arrays['max_features']]) or
        arrays['y_mean'].shape != (n_scales,) or arrays['coefficients'].shape != (n, n_scales) or
        arrays['reference_z'].ndim != 2 or arrays['reference_z'].shape[1] != n or
        arrays['reference_z'].shape[0] < 10 or
        any(not np.isfinite(arr).all() for arr in arrays.values()) or
        np.any(arrays['x_scale'] <= 0) or
        np.any(arrays['min_features'] > arrays['max_features'])):
        raise ValueError('Invalid model array shapes or values')
    if not np.isfinite(metadata['alpha']) or metadata['alpha'] <= 0 or not np.isfinite(metadata['max_seconds']) or metadata['max_seconds'] <= 0:
        raise ValueError('Invalid model configuration')
    return WavSemanticPredictor(tuple(scales), arrays['x_mean'], arrays['x_scale'],
            arrays['y_mean'], arrays['coefficients'], arrays['reference_z'],
            arrays['min_features'], arrays['max_features'], float(metadata['alpha']),
            float(metadata['max_seconds']), metadata['validation'])
