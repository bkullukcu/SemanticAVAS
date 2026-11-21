"""Participant-balanced scene-design regression for AVAS listening tests.

The predicted target is a scene's listener-mean score, not an individual
response. Design-factor prediction is separate from waveform prediction.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

SCENE_FEATURES = (
    'log1p_detuning_cents', 'detuning_not_specified', 'offset_seconds',
    'diagonal_geometry', 'single_source', 'identity_pair', 'route_variant_b',
)
DEFAULT_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)


def prepare_scene_ratings(trials, scales, *, participant='participant_id',
                          scene='scene_id', family='design_family'):
    """Balance repeated ratings within each participant/scene pair."""
    frame = pd.DataFrame(trials).copy()
    scales = tuple(scales)
    required = {participant, scene, family, 'scene_type', 'detuning_cents',
                'offset_s', 'geometry', 'spatial_variant', *scales}
    if not scales or len(set(scales)) != len(scales) or not required.issubset(frame):
        raise ValueError('Missing scene metadata, participant identifier or rating scales')
    for col in (participant, scene, family, 'scene_type', 'geometry', 'spatial_variant'):
        if frame[col].isna().any() or (frame[col].astype(str).str.strip() == '').any():
            raise ValueError(f'{col} contains missing identifiers or metadata')
    if not frame['scene_type'].isin(['single', 'pair', 'identity_pair']).all():
        raise ValueError('Only explicitly coded single and two-source scenes are supported')
    if not frame['geometry'].isin(['G0_LEFT_RIGHT', 'G45_DIAGONAL']).all():
        raise ValueError('Unknown geometry')
    if not frame['spatial_variant'].isin(['single', 'A', 'B']).all():
        raise ValueError('Unknown spatial variant')
    for scale in scales:
        frame[scale] = pd.to_numeric(frame[scale], errors='raise')
        if not np.isfinite(frame[scale]).all() or not frame[scale].between(0, 100).all():
            raise ValueError(f'{scale} must contain finite ratings in the range 0–100')
    raw_detuning = frame['detuning_cents']
    frame['detuning_cents'] = pd.to_numeric(raw_detuning, errors='coerce')
    if frame['detuning_cents'].isna().sum() != raw_detuning.isna().sum():
        raise ValueError('Invalid nonnumeric detuning_cents')
    frame['offset_s'] = pd.to_numeric(frame['offset_s'], errors='raise')
    if not np.isfinite(frame['offset_s']).all() or (frame['offset_s'] < 0).any():
        raise ValueError('offset_s requires nonnegative finite values')
    if ((frame['detuning_cents'] < 0) | np.isinf(frame['detuning_cents'])).any():
        raise ValueError('detuning_cents requires nonnegative finite values or NA')
    metadata = [family, 'scene_type', 'detuning_cents', 'offset_s',
                'geometry', 'spatial_variant']
    for col in metadata:
        if frame.groupby(scene, dropna=False)[col].nunique(dropna=False).gt(1).any():
            raise ValueError(f'{col} is inconsistent within a scene')
    balanced = frame.groupby([scene, participant], sort=True, as_index=False)[list(scales)].mean()
    counts = balanced.groupby(scene)[participant].nunique().sort_index()
    if len(counts) < 8 or (counts < 2).any():
        raise ValueError('At least eight scenes and two independent listeners per scene are required')
    per_scene = balanced.groupby(scene, sort=True)[list(scales)].mean()
    context = frame.drop_duplicates(scene).set_index(scene).loc[per_scene.index]
    detuning = context['detuning_cents'].to_numpy(float)
    missing = np.isnan(detuning)
    x = np.column_stack([
        np.log1p(np.nan_to_num(detuning, nan=0.0)), missing.astype(float),
        context['offset_s'].to_numpy(float),
        context['geometry'].eq('G45_DIAGONAL').to_numpy(float),
        context['scene_type'].eq('single').to_numpy(float),
        context['scene_type'].eq('identity_pair').to_numpy(float),
        context['spatial_variant'].eq('B').to_numpy(float),
    ])
    groups = context[family].astype(str).to_numpy()
    if len(set(groups)) < 3:
        raise ValueError('At least three scene-design families are needed for blocked validation')
    return x, per_scene.to_numpy(float), per_scene.index.to_numpy(str), groups, counts.to_numpy(int)


def _ridge(train_x, train_y, test_x, alpha):
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-12] = 1.0
    z = (train_x - mean) / std
    tz = (test_x - mean) / std
    ym = train_y.mean(axis=0)
    beta = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (train_y - ym))
    return np.clip(tz @ beta + ym, 0, 100)


def _folds(n, groups=None):
    if groups is None:
        return [np.array([i], dtype=int) for i in range(n)]
    return [np.flatnonzero(groups == label) for label in np.unique(groups)]


def nested_scene_cv(x, y, groups=None, *, alphas=DEFAULT_ALPHAS):
    """Nested sound or scene-family holdout with fold-local standardization."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 8:
        raise ValueError('Expected >=8 scenes with matching feature and rating matrices')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.logical_and(y >= 0, y <= 100).all():
        raise ValueError('Finite feature and 0–100 rating matrices are required')
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape != (len(x),) or pd.isna(groups).any() or len(set(groups)) < 4:
            raise ValueError('At least four complete scene-family labels required for nested family holdout')
    alphas = tuple(float(a) for a in alphas)
    if not alphas or any(not np.isfinite(a) or a <= 0 for a in alphas):
        raise ValueError('Ridge regularization values must be positive and finite')
    oof = np.empty_like(y)
    baseline = np.empty_like(y)
    selected = np.empty(len(x))
    indices = np.arange(len(x))
    for test in _folds(len(x), groups):
        train = np.setdiff1d(indices, test)
        inner_groups = None if groups is None else groups[train]
        inner = _folds(len(train), inner_groups)
        losses = []
        for alpha in alphas:
            errors = []
            for v in inner:
                t = np.setdiff1d(np.arange(len(train)), v)
                if len(t) < 3:
                    raise ValueError('Too few training scenes in an inner fold')
                errors.extend(np.abs(_ridge(x[train[t]], y[train[t]], x[train[v]], alpha) - y[train[v]]).mean(axis=1))
            losses.append(float(np.mean(errors)))
        best = alphas[int(np.argmin(losses))]
        oof[test] = _ridge(x[train], y[train], x[test], best)
        baseline[test] = y[train].mean(axis=0)
        selected[test] = best
    return oof, baseline, selected
