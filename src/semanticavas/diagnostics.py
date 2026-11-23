"""Per-source error analysis and participant-level measurement reliability.

Only reads caller-provided tables; no study-specific data are embedded here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _mean_error(frame, scale, kind):
    return np.abs(frame[f'{kind}_{scale}'].to_numpy(float)
                  - frame[f'observed_{scale}'].to_numpy(float))


def audit_predictions(frame: pd.DataFrame, scales, *, n_boot=2000, seed=2026):
    """Summarize out-of-fold errors by semantic scale and held-out source.

    Paired resampling uses entire source groups when available, not rows of
    the same source as independent units. Intervals are descriptive.
    """
    scales = tuple(scales)
    if len(scales) == 0 or len(set(scales)) != len(scales):
        raise ValueError('Unique named scales required')
    required = {'feature_set', 'model', 'holdout', 'sample_index', 'source_group'}
    required |= {f'{prefix}_{s}' for s in scales
                 for prefix in ('observed', 'predicted', 'baseline')}
    if not required.issubset(frame.columns):
        raise ValueError('Missing prediction, source or scale columns')
    if not isinstance(n_boot, int) or n_boot < 10:
        raise ValueError('n_boot must be at least 10')
    if frame.empty:
        raise ValueError('No held-out predictions supplied')
    group_summary, scale_summary, source_summary = [], [], []
    rng = np.random.default_rng(seed)
    keys = ['holdout', 'feature_set', 'model']
    for (holdout, features, model), piece in frame.groupby(keys, sort=True):
        if piece.sample_index.duplicated().any():
            raise ValueError('Duplicate test-sound rows within a model and protocol')
        observations = piece[[f'{kind}_{s}' for s in scales
                              for kind in ('observed', 'predicted', 'baseline')]]
        if not np.isfinite(observations.to_numpy(float)).all():
            raise ValueError('Predictions must be finite')
        err = np.column_stack([_mean_error(piece, s, 'predicted') for s in scales])
        ref = np.column_stack([_mean_error(piece, s, 'baseline') for s in scales])
        key = {'holdout': holdout, 'feature_set': features, 'model': model}
        for j, s in enumerate(scales):
            observed = piece[f'observed_{s}'].to_numpy(float)
            estimated = piece[f'predicted_{s}'].to_numpy(float)
            ss_total = np.sum((observed - observed.mean())**2)
            r2 = (float(1. - np.sum((estimated - observed)**2) / ss_total)
                  if ss_total > 0 else np.nan)
            scale_summary.append({**key, 'scale': s, 'n_sounds': len(piece),
                'mae': float(err[:, j].mean()),
                'baseline_mae': float(ref[:, j].mean()),
                'bias': float((estimated - observed).mean()),
                'r2': r2})
        sources = piece.source_group.fillna('').astype(str).to_numpy()
        split_by_source = bool(np.all(sources != ''))
        cluster = sources if split_by_source else piece.sample_index.astype(str).to_numpy()
        unique = np.unique(cluster)
        grouped_delta = []
        for source in unique:
            ix = cluster == source
            delta = float((err[ix] - ref[ix]).mean())
            grouped_delta.append(delta)
            if split_by_source:
                source_summary.append({**key, 'source_group': source,
                    'n_sounds': int(ix.sum()), 'mae': float(err[ix].mean()),
                    'baseline_mae': float(ref[ix].mean()),
                    'delta_mae': delta})
        # Cluster bootstrap of paired differences: no row-level resampling
        units = np.asarray(grouped_delta)
        draws = rng.integers(0, len(units), size=(n_boot, len(units)))
        replicated = units[draws].mean(axis=1)
        lo, hi = np.quantile(replicated, [.025, .975])
        group_summary.append({**key, 'n_sounds': len(piece),
            'n_resampling_units': len(unique),
            'resampling_unit': 'source_group' if split_by_source else 'sound',
            'sound_weighted_mae': float(err.mean()),
            'sound_weighted_baseline_mae': float(ref.mean()),
            'sound_weighted_delta_mae': float((err-ref).mean()),
            'unit_balanced_delta_mae': float(units.mean()),
            'unit_balanced_delta_ci_low': float(lo),
            'unit_balanced_delta_ci_high': float(hi),
            'ci_interpretation': 'descriptive paired percentile interval'})
    return (pd.DataFrame(group_summary), pd.DataFrame(scale_summary),
            pd.DataFrame(source_summary))


def listener_variability(ratings: pd.DataFrame, scales, *, n_splits=200, seed=2026):
    """Calculate listener disagreement from observed participant/sound responses.

    Repeated trials are averaged within each (participant, stimulus) before
    deriving any mean, uncertainty estimate or split-half reliability.
    """
    scales = tuple(scales)
    if len(scales) == 0 or len(set(scales)) != len(scales):
        raise ValueError('Unique scales required')
    if not {'participant', 'stimulus', *scales}.issubset(ratings.columns):
        raise ValueError('Missing participant, stimulus or scale column')
    if ratings[['participant', 'stimulus']].isna().any().any():
        raise ValueError('Listener and stimulus identifiers must be complete')
    f = ratings[['participant', 'stimulus', *scales]].copy()
    for s in scales:
        f[s] = pd.to_numeric(f[s], errors='raise')
        if np.isinf(f[s].to_numpy(float)).any():
            raise ValueError('Infinite rating values are unsupported')
    f = f.groupby(['participant', 'stimulus'], as_index=False)[list(scales)].mean()
    if f.participant.nunique() < 4 or f.stimulus.nunique() < 3:
        raise ValueError('At least four listeners and three sounds are needed')
    rows, reliability = [], []
    for stimulus, item in f.groupby('stimulus', sort=True):
        for s in scales:
            v = item[s].dropna().to_numpy(float)
            if len(v) < 2:
                continue
            sd = float(np.std(v, ddof=1))
            q25, q75 = np.quantile(v, [.25, .75])
            rows.append({'stimulus': stimulus, 'scale': s, 'n_listeners': len(v),
                'mean': float(np.mean(v)), 'sd': sd, 'sem': sd / np.sqrt(len(v)),
                'iqr': float(q75-q25)})
    rng = np.random.default_rng(seed)
    listeners = np.array(sorted(f.participant.unique(), key=str), dtype=object)
    for s in scales:
        correlations = []
        for _ in range(n_splits):
            permutation = rng.permutation(listeners)
            left, right = np.array_split(permutation, 2)
            a = f[f.participant.isin(left)].groupby('stimulus')[s].mean()
            b = f[f.participant.isin(right)].groupby('stimulus')[s].mean()
            paired = pd.concat([a, b], axis=1, join='inner').dropna().to_numpy(float)
            if len(paired) >= 3 and np.std(paired[:, 0]) > 0 and np.std(paired[:, 1]) > 0:
                correlations.append(float(np.corrcoef(paired.T)[0, 1]))
        reliability.append({'scale': s, 'split_half_mean_r': (
            float(np.mean(correlations)) if correlations else np.nan),
            'valid_splits': len(correlations),
            'n_requested_splits': n_splits})
    return pd.DataFrame(rows), pd.DataFrame(reliability)


def main(argv=None):
    p = argparse.ArgumentParser(description='Inspect held-out errors and listener disagreement')
    p.add_argument('--predictions', required=True)
    p.add_argument('--ratings', help='Optional observed participant ratings CSV')
    p.add_argument('--scales', nargs='+', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--bootstrap', type=int, default=2000)
    args = p.parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_model, by_scale, by_source = audit_predictions(
        pd.read_csv(args.predictions), args.scales, n_boot=args.bootstrap)
    for name, table in [('models', by_model), ('scales', by_scale),
                        ('sources', by_source)]:
        table.to_csv(out / f'{name}.csv', index=False)
    if args.ratings:
        per_sound, reliability = listener_variability(
            pd.read_csv(args.ratings), args.scales)
        per_sound.to_csv(out / 'listener_variability.csv', index=False)
        reliability.to_csv(out / 'listener_split_half.csv', index=False)
    (out / 'method.json').write_text(json.dumps({
        'paired_bootstrap_units': 'source when declared; otherwise sound',
        'interval': 'percentile, descriptive',
        'repeat_policy': 'average within listener and stimulus',
        'privacy': 'derived files may contain restricted identifiers'
    }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
