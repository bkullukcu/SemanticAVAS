"""Generated-data regression checks for exploratory acoustic comparisons."""
import numpy as np
from scipy.io import wavfile
from semanticavas.acoustic_features import FEATURE_NAMES
from semanticavas.feature_ablation import (
    EXTRA_NAMES, FEATURE_SETS, extract_experimental_features,
    nested_ablation, summarize_rows,
)


def test_tonal_features_distinguish_tone_and_noise(tmp_path):
    fs = 16000
    t = np.arange(fs * 2) / fs
    tone = (.25 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    noise = (.25 * np.random.default_rng(13).normal(size=len(t))).astype(np.float32)
    wavfile.write(tmp_path / 'tone.wav', fs, tone)
    wavfile.write(tmp_path / 'noise.wav', fs, noise)
    a = extract_experimental_features(tmp_path / 'tone.wav')
    b = extract_experimental_features(tmp_path / 'noise.wav')
    assert set(a) == set(EXTRA_NAMES)
    assert a['dominant_tonal_fraction'] > b['dominant_tonal_fraction'] + .3
    assert np.isfinite(list(a.values())).all()


def test_amplitude_modulation_changes_temporal_feature(tmp_path):
    fs = 16000
    t = np.arange(fs * 3) / fs
    stationary = (.1 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
    modulated = (stationary * (1 + .8 * np.sin(2 * np.pi * 5 * t))).astype(np.float32)
    wavfile.write(tmp_path / 'stationary.wav', fs, stationary)
    wavfile.write(tmp_path / 'modulated.wav', fs, modulated)
    a = extract_experimental_features(tmp_path / 'stationary.wav')
    b = extract_experimental_features(tmp_path / 'modulated.wav')
    assert b['frame_rms_coefficient_of_variation'] > a['frame_rms_coefficient_of_variation'] + .25


def test_held_out_model_comparisons_use_matching_baselines():
    x = np.random.default_rng(7).normal(
        size=(12, len(FEATURE_NAMES) + len(EXTRA_NAMES)))
    y = np.column_stack([40 + 3 * x[:, 0], 50 + 2 * x[:, -1]])
    groups = np.repeat(['F1', 'F2', 'F3', 'F4'], 3)
    rows = nested_ablation(x, y, groups=groups, models=('ridge',))
    predictions, summary = summarize_rows(rows, ['truck_like', 'annoying'])
    assert set(predictions.feature_set) == set(FEATURE_SETS)
    assert len(predictions) == len(x) * len(FEATURE_SETS)
    assert np.isfinite(summary.mae).all()
    for sample, subset in predictions.groupby('sample_index'):
        for scale in ('truck_like', 'annoying'):
            assert subset[f'baseline_{scale}'].nunique() == 1
        assert set(subset.source_group) == {groups[sample]}


def test_single_source_group_is_rejected():
    x = np.ones((12, len(FEATURE_NAMES) + len(EXTRA_NAMES)))
    y = np.full((12, 1), 40.)
    try:
        nested_ablation(x, y, groups=['same'] * 12, models=('ridge',))
    except ValueError as exc:
        assert 'three declared groups' in str(exc)
    else:
        raise AssertionError('Invalid source grouping accepted')
