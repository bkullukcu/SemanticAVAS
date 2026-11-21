"""Synthetic validation contracts for listener panels and semantic scales.

These checks exercise existing input validation; no listener observations are used.
"""
import numpy as np
import pandas as pd
import pytest

from semanticavas.empirical import (
    aggregate_semantics,
    bootstrap_profiles,
    validate_semantic_ratings,
)


def _panel():
    return pd.DataFrame([
        {"participant": person, "stimulus": sound, "pleasant": value}
        for person, values in (
            ("P1", (20.0, 80.0)),
            ("P2", (40.0, 60.0)),
            ("P3", (60.0, 40.0)),
        )
        for sound, value in zip(("A", "B"), values)
    ])


def test_missing_listener_stimulus_cell_is_not_a_complete_panel():
    incomplete = _panel().iloc[:-1].copy()
    with pytest.raises(ValueError, match="incomplete"):
        validate_semantic_ratings(incomplete, ["pleasant"])
    with pytest.raises(ValueError, match="incomplete"):
        aggregate_semantics(incomplete, ["pleasant"])


@pytest.mark.parametrize("bad", [-0.01, 100.01, np.inf, -np.inf])
def test_out_of_range_or_nonfinite_semantic_rating_is_rejected(bad):
    frame = _panel()
    frame.loc[0, "pleasant"] = bad
    with pytest.raises(ValueError, match="outside 0-100 or nonfinite"):
        validate_semantic_ratings(frame, ["pleasant"])


def test_bootstrap_listener_clusters_reproducibly_and_keeps_valid_bounds():
    frame = _panel()
    lower_a, upper_a = bootstrap_profiles(
        frame, ["pleasant"], n_boot=200, seed=21
    )
    lower_b, upper_b = bootstrap_profiles(
        frame, ["pleasant"], n_boot=200, seed=21
    )
    pd.testing.assert_frame_equal(lower_a, lower_b)
    pd.testing.assert_frame_equal(upper_a, upper_b)
    assert lower_a.index.tolist() == ["A", "B"]
    assert lower_a.columns.tolist() == ["pleasant"]
    assert (lower_a.to_numpy() <= upper_a.to_numpy()).all()
    assert (lower_a.to_numpy() >= 20).all()
    assert (upper_a.to_numpy() <= 80).all()
