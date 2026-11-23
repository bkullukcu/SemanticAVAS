"""Exploratory tonal/temporal features and group-held-out model comparisons.

This research module does not modify the released ten-descriptor predictor.
Only recording-derived features are computed; listening data stay private.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import find_peaks, stft, welch

from .acoustic_features import FEATURE_NAMES
from .prediction import prepare_training

TONAL_NAMES = ("spectral_peak_local_excess_db", "dominant_tonal_fraction")
TEMPORAL_NAMES = ("frame_rms_coefficient_of_variation", "mean_positive_spectral_flux")
EXTRA_NAMES = TONAL_NAMES + TEMPORAL_NAMES
FEATURE_SETS = {"E0": (), "E1": TONAL_NAMES, "E2": TEMPORAL_NAMES}
DEFAULT_GRIDS = {"ridge": (0.1, 1., 10., 100., 1000.), "svr": (0.1, 1., 10.), "pls": (1, 2, 3)}


def _convert_pcm(array):
    x = array.astype(np.float64)
    if array.dtype.kind == 'u':
        midpoint = (np.iinfo(array.dtype).max + 1) / 2
        x = (x - midpoint) / midpoint
    elif array.dtype.kind == 'i':
        limit = max(abs(np.iinfo(array.dtype).min), np.iinfo(array.dtype).max)
        x = x / limit
    if not np.isfinite(x).all():
        raise ValueError('WAV contains non-finite samples')
    return x


def extract_experimental_features(path: str | Path, *, max_seconds: float | None = 5.0):
    """Two tonal and two temporal DIGITAL-SIGNAL descriptors.

    Uses 50 ms RMS frames and 64 ms spectral-flux windows.
    This is not a loudness, tonality-standard or calibrated-SPL estimator.
    """
    path = Path(path)
    if path.suffix.lower() != '.wav' or not path.is_file():
        raise ValueError('An existing .wav path is required')
    if path.stat().st_size > 100_000_000:
        raise ValueError('WAV exceeds 100 MB input limit')
    if max_seconds is not None and (not np.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError('max_seconds must be positive or None')
    fs, pcm = wavfile.read(BytesIO(path.read_bytes()))
    if fs < 8000 or pcm.ndim not in (1, 2):
        raise ValueError('Unsupported PCM WAV sampling rate or channel layout')
    x = _convert_pcm(pcm)
    if max_seconds is not None:
        x = x[:int(max_seconds * fs)]
    if x.ndim == 1:
        x = x[:, None]
    if len(x) < 4096:
        raise ValueError('WAV requires at least 4096 samples')
    channel_results = []
    for ch in x.T:
        rms = float(np.sqrt(np.mean(ch**2)))
        if rms <= 1e-12:
            continue
        segment = min(4096, len(ch))
        frequencies, power = welch(ch, fs=fs, nperseg=segment,
                                   noverlap=segment // 2, window='hann',
                                   detrend='constant', scaling='density')
        valid = (frequencies >= 80.) & (frequencies <= min(10000., fs / 2.))
        f, p = frequencies[valid], power[valid]
        if len(p) < 8 or p.sum() <= 0:
            raise ValueError('Insufficient power within analysis band')
        peaks, _ = find_peaks(p)
        best = int(peaks[np.argmax(p[peaks])]) if len(peaks) else int(np.argmax(p))
        distance = np.abs(f - f[best])
        background = p[(distance >= 30.) & (distance <= 180.)]
        if len(background) < 3:
            background = np.delete(p, best)
        local_background = float(np.median(background))
        excess = float(np.clip(
            10. * np.log10((p[best] + 1e-20) / (local_background + 1e-20)), 0., 100.))
        concentration = float(p[distance <= 25.].sum() / p.sum())
        frame = max(64, int(round(fs * .050)))
        count = len(ch) // frame
        frame_rms = np.sqrt(np.mean(ch[:count * frame].reshape(count, frame)**2, axis=1))
        rms_cv = float(np.std(frame_rms) / max(float(np.mean(frame_rms)), 1e-12))
        window = max(256, int(round(fs * .064)))
        _, _, coeff = stft(ch, fs=fs, nperseg=window,
                           noverlap=window // 2, boundary=None, padded=False)
        magnitudes = np.abs(coeff)
        normalized = magnitudes / np.maximum(magnitudes.sum(axis=0, keepdims=True), 1e-12)
        flux = (float(np.mean(np.maximum(np.diff(normalized, axis=1), 0.).sum(axis=0)))
                if normalized.shape[1] > 1 else 0.)
        channel_results.append((excess, concentration, rms_cv, flux))
    if not channel_results:
        raise ValueError('WAV contains no non-silent channels')
    mean = np.mean(channel_results, axis=0)
    if not np.isfinite(mean).all():
        raise ValueError('Non-finite derived signal descriptors')
    return dict(zip(EXTRA_NAMES, map(float, mean)))


def _model(name, parameter):
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.linear_model import Ridge
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    if name == 'ridge':
        return make_pipeline(StandardScaler(), Ridge(alpha=float(parameter)))
    if name == 'svr':
        return make_pipeline(StandardScaler(), MultiOutputRegressor(
            SVR(C=float(parameter), gamma='scale')))
    if name == 'pls':
        return make_pipeline(StandardScaler(), PLSRegression(
            n_components=int(parameter), scale=False))
    raise ValueError('Unknown model: ' + name)


def _inner_folds(indices, groups):
    if groups is None:
        return [part for part in np.array_split(indices, min(4, len(indices))) if len(part)]
    return [indices[groups[indices] == key] for key in np.unique(groups[indices])]


def nested_ablation(x, y, *, groups=None, models=('ridge', 'svr', 'pls'),
                    lower=None, upper=None):
    """Evaluate E0/E1/E2 with identical outer splits and training-only fitting."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 10:
        raise ValueError('At least ten aligned sound/target records are required')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Numeric features and targets must be finite')
    if x.shape[1] != len(FEATURE_NAMES) + len(EXTRA_NAMES):
        raise ValueError('Input must contain E0 followed by four experimental descriptors')
    if groups is not None:
        groups = np.asarray(groups, dtype=str)
        if len(groups) != len(x) or len(np.unique(groups)) < 3 or np.any(groups == ''):
            raise ValueError('Source validation requires at least three declared groups')
    lower = np.zeros(y.shape[1]) if lower is None else np.asarray(lower, float)
    upper = np.full(y.shape[1], 100.) if upper is None else np.asarray(upper, float)
    if (lower.shape != (y.shape[1],) or upper.shape != (y.shape[1],)
            or not np.isfinite(lower).all() or not np.isfinite(upper).all()
            or np.any(lower >= upper) or np.any((y < lower) | (y > upper))):
        raise ValueError('Invalid or inconsistent questionnaire bounds')
    keys = {name: i for i, name in enumerate(tuple(FEATURE_NAMES) + EXTRA_NAMES)}
    feature_matrices = {
        experiment: x[:, [keys[name] for name in tuple(FEATURE_NAMES) + extra]]
        for experiment, extra in FEATURE_SETS.items()
    }
    all_indices = np.arange(len(x))
    outer = ([np.array([i]) for i in all_indices] if groups is None else
             [np.flatnonzero(groups == label) for label in np.unique(groups)])
    rows = []
    for experiment, matrix in feature_matrices.items():
        for model_name in models:
            if model_name not in DEFAULT_GRIDS:
                raise ValueError('Model must be ridge, svr or pls')
            for test in outer:
                train = np.setdiff1d(all_indices, test)
                inner = _inner_folds(train, groups)
                scores = []
                for parameter in DEFAULT_GRIDS[model_name]:
                    errors = []
                    for validation in inner:
                        fit = np.setdiff1d(train, validation)
                        if (len(fit) < 3 or
                                (model_name == 'pls' and parameter > min(len(fit) - 1, matrix.shape[1]))):
                            errors.append(np.inf)
                            continue
                        fitted = _model(model_name, parameter)
                        fitted.fit(matrix[fit], y[fit])
                        predicted = np.clip(np.atleast_2d(fitted.predict(matrix[validation])), lower, upper)
                        errors.append(float(np.abs(predicted - y[validation]).mean()))
                    scores.append(float(np.mean(errors)))
                if not np.isfinite(scores).any():
                    raise ValueError('Insufficient training observations for inner tuning')
                selected = DEFAULT_GRIDS[model_name][int(np.argmin(scores))]
                fitted = _model(model_name, selected)
                fitted.fit(matrix[train], y[train])
                predicted = np.clip(np.atleast_2d(fitted.predict(matrix[test])), lower, upper)
                baseline = y[train].mean(axis=0)
                for position, sample in enumerate(test):
                    rows.append((experiment, model_name, int(sample),
                                 str(groups[sample]) if groups is not None else '',
                                 float(selected), y[sample].copy(),
                                 predicted[position].copy(), baseline.copy()))
    return rows


def summarize_rows(rows, scales):
    """Collect per-scale predictions, sound-weighted and source-balanced MAE."""
    records, summary = [], []
    for family, model, index, source, parameter, actual, predicted, baseline in rows:
        row = {'feature_set': family, 'model': model, 'sample_index': index,
               'source_group': source, 'chosen_hyperparameter': parameter}
        for j, scale in enumerate(scales):
            row[f'observed_{scale}'] = float(actual[j])
            row[f'predicted_{scale}'] = float(predicted[j])
            row[f'baseline_{scale}'] = float(baseline[j])
        records.append(row)
    predictions = pd.DataFrame(records)
    for (family, model), subset in predictions.groupby(['feature_set', 'model'], sort=True):
        errors = np.column_stack([
            abs(subset[f'predicted_{scale}'] - subset[f'observed_{scale}']) for scale in scales])
        reference = np.column_stack([
            abs(subset[f'baseline_{scale}'] - subset[f'observed_{scale}']) for scale in scales])
        result = {'feature_set': family, 'model': model, 'n_sounds': len(subset),
                  'mae': float(errors.mean()),
                  'mean_baseline_mae': float(reference.mean()),
                  'delta_mae': float(errors.mean() - reference.mean())}
        for j, scale in enumerate(scales):
            result[f'mae_{scale}'] = float(errors[:, j].mean())
        if subset['source_group'].astype(bool).all():
            grouped = pd.DataFrame({'source': subset['source_group'],
                                    'error': errors.mean(axis=1)})
            result['equal_source_mean_mae'] = float(
                grouped.groupby('source')['error'].mean().mean())
        summary.append(result)
    return predictions, pd.DataFrame(summary)


def run(manifest_path, ratings_path, scales, output_dir, *,
        models=('ridge', 'svr', 'pls'), max_seconds=5., scale_bounds=None):
    """Read consent-restricted WAV/rating pairs without committing their contents."""
    manifest_path = Path(manifest_path)
    baseline, y, groups, audio = prepare_training(
        manifest_path, ratings_path, scales,
        max_seconds=max_seconds, scale_bounds=scale_bounds)
    manifest = pd.read_csv(manifest_path, dtype={'stimulus': str, 'wav': str})
    file_map = manifest.set_index('stimulus')['wav'].to_dict()
    additional, audit = [], []
    for index, stimulus in enumerate(audio['stimulus']):
        wav_path = Path(file_map[stimulus])
        if not wav_path.is_absolute():
            wav_path = manifest_path.resolve().parent / wav_path
        descriptors = extract_experimental_features(wav_path, max_seconds=max_seconds)
        additional.append([descriptors[name] for name in EXTRA_NAMES])
        audit.append({'stimulus': stimulus,
                      'sound_family': '' if groups is None else groups[index],
                      'wav_sha256': sha256(wav_path.read_bytes()).hexdigest(),
                      **descriptors})
    x = np.column_stack((baseline, np.asarray(additional, float)))
    limits = {name: (0., 100.) if scale_bounds is None else scale_bounds[name]
              for name in scales}
    lower = np.array([limits[name][0] for name in scales], float)
    upper = np.array([limits[name][1] for name in scales], float)
    sound_rows = nested_ablation(x, y, models=models, lower=lower, upper=upper)
    sound_predictions, sound_summary = summarize_rows(sound_rows, scales)
    sound_predictions['stimulus'] = sound_predictions['sample_index'].map(audio['stimulus'])
    sound_summary['holdout'] = 'one_sound'
    predictions = [sound_predictions.assign(holdout='one_sound')]
    summaries = [sound_summary]
    if groups is not None:
        source_rows = nested_ablation(x, y, groups=groups, models=models,
                                     lower=lower, upper=upper)
        source_predictions, source_summary = summarize_rows(source_rows, scales)
        source_predictions['stimulus'] = source_predictions['sample_index'].map(audio['stimulus'])
        source_summary['holdout'] = 'one_source_group'
        predictions.append(source_predictions.assign(holdout='one_source_group'))
        summaries.append(source_summary)
    result = pd.concat(summaries, ignore_index=True)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    pd.concat(predictions, ignore_index=True).to_csv(
        destination / 'outer_predictions.csv', index=False)
    result.to_csv(destination / 'comparison.csv', index=False)
    pd.DataFrame(audit).to_csv(destination / 'feature_provenance.csv', index=False)
    protocol = {'status': 'exploratory', 'n_sounds': int(len(y)),
                'n_scales': len(scales),
                'source_groups': None if groups is None else int(len(np.unique(groups))),
                'feature_sets': {n: list(FEATURE_NAMES) + list(v)
                                 for n, v in FEATURE_SETS.items()},
                'models': list(models),
                'guidance': ('Source-group validation is required for cross-source claims. '
                             'Do not select a winner on the same test blocks.')}
    (destination / 'protocol.json').write_text(
        json.dumps(protocol, indent=2), encoding='utf-8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description='Exploratory E0/E1/E2 feature comparison')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--ratings', required=True)
    parser.add_argument('--scales', nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--models', nargs='+', choices=list(DEFAULT_GRIDS),
                        default=list(DEFAULT_GRIDS))
    parser.add_argument('--max-seconds', type=float, default=5.)
    parser.add_argument('--full-recording', action='store_true')
    parser.add_argument('--scale-bounds', type=json.loads)
    arguments = parser.parse_args(argv)
    result = run(arguments.manifest, arguments.ratings, arguments.scales,
                 arguments.output_dir, models=arguments.models,
                 max_seconds=None if arguments.full_recording else arguments.max_seconds,
                 scale_bounds=arguments.scale_bounds)
    print(result[['holdout', 'feature_set', 'model',
                  'mae', 'mean_baseline_mae', 'delta_mae']].to_string(index=False))


if __name__ == '__main__':
    main()
