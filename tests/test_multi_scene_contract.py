"""Synthetic regression tests for scene-design and WAV provenance contracts."""
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile

from semanticavas.scene_model import prepare_scene_ratings, nested_scene_cv, SCENE_FEATURES
from semanticavas.prediction import prepare_training


def _scene_trials():
    rows = []
    for i in range(12):
        for listener in ('P1', 'P2', 'P3'):
            for repeat in range(2 if listener == 'P1' else 1):
                rows.append({
                    'participant_id': listener, 'scene_id': f'S{i:02d}',
                    'design_family': f'F{i // 3}',
                    'scene_type': 'single' if i % 3 == 0 else 'pair',
                    'detuning_cents': float(i * 25),
                    'offset_s': float(i % 2),
                    'geometry': 'G0_LEFT_RIGHT' if i % 2 else 'G45_DIAGONAL',
                    'spatial_variant': 'A' if i % 2 else 'B',
                    'rating': float(20 + 5*i + {'P1':2, 'P2':0, 'P3':-2}[listener] + (6 if repeat else 0))
                })
    return pd.DataFrame(rows)


def test_participant_balancing_and_nested_splits():
    x, y, ids, families, listener_counts = prepare_scene_ratings(_scene_trials(), ('rating',))
    assert x.shape == (12, len(SCENE_FEATURES))
    assert y.shape == (12, 1)
    assert np.all(listener_counts == 3)
    assert y[0, 0] == pytest.approx(21.0)
    for split in (None, families):
        fitted, mean, alphas = nested_scene_cv(x, y, split)
        assert fitted.shape == y.shape == mean.shape
        assert np.isfinite(fitted).all()
        assert all(a > 0 for a in alphas)


def test_repeated_trials_never_gain_listener_weight():
    frame = _scene_trials()
    original = prepare_scene_ratings(frame, ['rating'])[1][0, 0]
    duplicate = frame[(frame.scene_id == 'S00') & (frame.participant_id == 'P1')].iloc[[0]]
    amended = pd.concat([frame, duplicate], ignore_index=True)
    assert prepare_scene_ratings(amended, ['rating'])[1][0, 0] == pytest.approx(original - 1/3)


def test_scene_metadata_inconsistency_rejected():
    frame = _scene_trials()
    frame.loc[0, 'detuning_cents'] = 999
    with pytest.raises(ValueError, match='inconsistent'):
        prepare_scene_ratings(frame, ['rating'])


def test_scene_rating_range_and_invalid_family_rejected():
    frame = _scene_trials()
    frame.loc[0, 'rating'] = -1
    with pytest.raises(ValueError, match='0–100'):
        prepare_scene_ratings(frame, ['rating'])
    frame = _scene_trials()
    frame['design_family'] = 'only-one'
    with pytest.raises(ValueError, match='three'):
        prepare_scene_ratings(frame, ['rating'])


def test_optional_wav_sha256_contract(tmp_path):
    manifest, ratings = [], []
    fs = 16000
    t = np.arange(fs) / fs
    for i in range(10):
        path = tmp_path / f'S{i:02d}.wav'
        wavfile.write(path, fs, (0.12*np.sin(2*np.pi*(220+30*i)*t)).astype('float32'))
        manifest.append({'stimulus': f'S{i:02d}', 'wav': path.name,
                         'wav_sha256': sha256(path.read_bytes()).hexdigest()})
        for listener in ('a', 'b', 'c'):
            ratings.append({'participant':listener,'stimulus':f'S{i:02d}','score':float(i*8)})
    mp = tmp_path / 'manifest.csv'
    rp = tmp_path / 'ratings.csv'
    pd.DataFrame(manifest).to_csv(mp,index=False)
    pd.DataFrame(ratings).to_csv(rp,index=False)
    x, y, _, _ = prepare_training(mp,rp,['score'])
    assert x.shape[0] == y.shape[0] == 10
    manifest[0]['wav_sha256'] = 'f'*64
    pd.DataFrame(manifest).to_csv(mp,index=False)
    with pytest.raises(ValueError,match='SHA-256 mismatch'):
        prepare_training(mp,rp,['score'])


def test_fit_new_scene_inference():
    from semanticavas.scene_model import fit_scene_predictor, encode_scene_design
    frame = _scene_trials()
    x, y, _, families, _ = prepare_scene_ratings(frame, ['rating'])
    model = fit_scene_predictor(x, y, ['rating'], groups=families)
    new_scenes = frame.drop_duplicates('scene_id').iloc[:2]
    predicted = model.predict_scenes(new_scenes)
    assert list(predicted) == ['rating'] and predicted.shape == (2, 1)
    assert np.isfinite(predicted.values).all()
    assert np.all((predicted.values >= 0) & (predicted.values <= 100))
    assert np.allclose(predicted.values, model.predict_features(encode_scene_design(new_scenes)))


def test_scene_model_archive_roundtrip(tmp_path):
    from semanticavas.scene_model import (
        fit_scene_predictor, save_scene_predictor, load_scene_predictor)
    x, y, _, families, _ = prepare_scene_ratings(_scene_trials(), ['rating'])
    model = fit_scene_predictor(x, y, ['rating'], groups=families)
    path = save_scene_predictor(model, tmp_path / 'scene.npz')
    loaded = load_scene_predictor(path)
    assert loaded.scales == model.scales
    np.testing.assert_allclose(loaded.predict_features(x), model.predict_features(x))
