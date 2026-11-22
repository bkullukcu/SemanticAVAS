"""Deterministic contracts for listener balance and family-blocked CV.

Synthetic identifiers and scores are not experimental listener observations.
"""
import numpy as np
import pandas as pd

from semanticavas import empirical, prediction


def test_repeated_trial_keeps_each_listener_equally_weighted(tmp_path, monkeypatch):
    """A second trial from one listener must not count as a third listener."""
    manifest = pd.DataFrame({
        "stimulus": [f"S{i:02d}" for i in range(10)],
        "wav": [f"sound_{i}.wav" for i in range(10)],
        "sound_family": [f"F{i // 2}" for i in range(10)],
    })
    ratings = pd.DataFrame([
        {"stimulus": sound, "participant": participant, "pleasant": score}
        for sound in manifest["stimulus"]
        for participant, score in [("P1", 20.0), ("P2", 80.0)]
    ])
    manifest.to_csv(tmp_path / "manifest.csv", index=False)
    ratings.to_csv(tmp_path / "ratings.csv", index=False)

    def synthetic_features(path, *, max_seconds=5.0):
        number = int(path.stem.split("_")[1])
        return np.full(len(prediction.FEATURE_NAMES), number, dtype=float), {
            "sha256": f"{number:064x}",
            "sample_rate_hz": 16000,
            "duration_s": 1.0,
            "channels": 1,
        }

    monkeypatch.setattr(prediction, "extract_wav", synthetic_features)
    x_before, y_before, families_before, _ = prediction.prepare_training(
        tmp_path / "manifest.csv", tmp_path / "ratings.csv", ["pleasant"]
    )
    # Duplicate P1's trial for one sound: its rating must not gain weight.
    ratings = pd.concat([ratings, ratings.iloc[[0]]], ignore_index=True)
    ratings.to_csv(tmp_path / "ratings.csv", index=False)
    x_after, y_after, families_after, _ = prediction.prepare_training(
        tmp_path / "manifest.csv", tmp_path / "ratings.csv", ["pleasant"]
    )
    np.testing.assert_array_equal(x_before, x_after)
    np.testing.assert_array_equal(y_before, y_after)
    np.testing.assert_array_equal(y_after, np.full((10, 1), 50.0))
    np.testing.assert_array_equal(families_before, families_after)


def test_nested_family_cv_never_trains_on_held_out_family(monkeypatch):
    """Every ridge fit, including inner tuning, separates entire families."""
    x = np.column_stack([np.arange(12, dtype=float), np.arange(12, dtype=float) ** 2])
    y = (x[:, :1] * 3 + 10).astype(float)
    groups = np.repeat(["A", "B", "C", "D"], 3)
    observed = []

    def checked_ridge(train_x, train_y, test_x, alpha):
        train_groups = set(groups[train_x[:, 0].astype(int)])
        test_groups = set(groups[test_x[:, 0].astype(int)])
        assert train_groups.isdisjoint(test_groups)
        observed.append((frozenset(train_groups), frozenset(test_groups)))
        return np.full((len(test_x), train_y.shape[1]), train_y.mean(axis=0))

    monkeypatch.setattr(empirical, "_ridge_fit_predict", checked_ridge)
    predicted, baseline, chosen = empirical.nested_leave_one_group_out(
        x, y, groups, alpha_grid=(0.1, 1.0)
    )
    assert len(observed) > 4
    assert predicted.shape == baseline.shape == (12, 1)
    assert chosen.shape == (12,)
    assert np.isfinite(predicted).all()
    assert set(np.unique(chosen)).issubset({0.1, 1.0})
