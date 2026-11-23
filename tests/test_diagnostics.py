"""Generated-data tests for held-out prediction audits and listener summaries."""
import numpy as np
import pandas as pd
import pytest
from semanticavas.diagnostics import audit_predictions, listener_variability


def _predictions():
    rows = []
    for i in range(12):
        rows.append({'feature_set': 'E0', 'model': 'ridge', 'holdout': 'source',
            'sample_index': i, 'source_group': f'G{i//3}',
            'observed_a': 30 + i, 'predicted_a': 31 + i,
            'baseline_a': 32 + i, 'observed_b': 40 + i,
            'predicted_b': 40 + i, 'baseline_b': 42 + i})
    return pd.DataFrame(rows)


def test_source_group_and_scale_errors_are_deterministic():
    model, scale, source = audit_predictions(_predictions(), ['a','b'], n_boot=100, seed=7)
    assert model.iloc[0].n_resampling_units == 4
    assert model.iloc[0].resampling_unit == 'source_group'
    assert len(source) == 4 and len(scale) == 2
    assert model.iloc[0].sound_weighted_delta_mae == pytest.approx(-1.5)
    assert model.iloc[0].unit_balanced_delta_ci_low == pytest.approx(-1.5)


def test_shuffled_rows_do_not_change_overall_errors():
    frame = _predictions()
    a, _, _ = audit_predictions(frame, ['a','b'], n_boot=50)
    b, _, _ = audit_predictions(frame.sample(frac=1, random_state=1), ['a','b'], n_boot=50)
    assert a.iloc[0].sound_weighted_mae == pytest.approx(b.iloc[0].sound_weighted_mae)


def test_repeated_trials_are_balanced_before_sem():
    f = pd.DataFrame([
        {'participant': p, 'stimulus': s, 'pleasant': v}
        for s in ['S1','S2','S3','S4'] for p,v in [
            ('a',10),('a',30),('b',20),('c',50),('d',40)]
    ])
    per_sound, reliability = listener_variability(f, ['pleasant'], n_splits=20)
    assert per_sound.n_listeners.min() == 4
    assert per_sound.iloc[0]['mean'] == pytest.approx(32.5)
    assert reliability.iloc[0].valid_splits == 0  # no between-stimulus variation


def test_missing_columns_are_rejected():
    with pytest.raises(ValueError, match='Missing prediction'):
        audit_predictions(pd.DataFrame({'sample_index':[0]}), ['a'])


def test_mixed_effects_requires_sufficient_independent_listeners():
    from semanticavas.diagnostics import fit_listener_random_intercepts
    f=pd.DataFrame({'participant':['a','b','c'],'stimulus':['S1']*3,
                    'pleasant':[20,30,40]})
    with pytest.raises(ValueError,match='at least four'):
        fit_listener_random_intercepts(f,['pleasant'])
