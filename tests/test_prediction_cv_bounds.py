"""Regression checks for scale-bounded nested evaluation (synthetic values)."""
import numpy as np
import pytest

import semanticavas.empirical as empirical
import semanticavas.prediction as prediction


def _artificial_prediction(train_x, train_y, test_x, alpha):
    # The low-penalty candidate is outside the 1-5 questionnaire scale.
    # Clipping changes the correct choice of penalty in this fixture.
    return np.full((len(test_x), train_y.shape[1]), 50. if alpha == .1 else 4.)


def test_final_alpha_selection_honours_declared_bounds(monkeypatch):
    monkeypatch.setattr(prediction, '_ridge_fit_predict', _artificial_prediction)
    x = np.arange(12.).reshape(-1, 1)
    y = np.full((12, 1), 5.)
    assert prediction._choose_alpha(x, y, None, (.1, 1.), (np.array([1.]), np.array([5.]))) == .1
    assert prediction._choose_alpha(x, y, None, (.1, 1.), (np.array([0.]), np.array([100.]))) == 1.


def test_sound_holdout_uses_bounds_for_tuning_and_predictions(monkeypatch):
    monkeypatch.setattr(empirical, '_ridge_fit_predict', _artificial_prediction)
    x = np.arange(12.).reshape(-1, 1)
    y = np.full((12, 1), 5.)
    pred, _, chosen = empirical.nested_leave_one_sound_out(
        x, y, alpha_grid=(.1, 1.), prediction_bounds=(np.array([1.]), np.array([5.])))
    np.testing.assert_allclose(pred, 5.)
    np.testing.assert_allclose(chosen, .1)


def test_group_holdout_uses_bounds_for_tuning_and_predictions(monkeypatch):
    monkeypatch.setattr(empirical, '_ridge_fit_predict', _artificial_prediction)
    x = np.arange(12.).reshape(-1, 1)
    y = np.full((12, 1), 5.)
    groups = np.repeat(['a', 'b', 'c', 'd'], 3)
    pred, _, chosen = empirical.nested_leave_one_group_out(
        x, y, groups, alpha_grid=(.1, 1.), prediction_bounds=(np.array([1.]), np.array([5.])))
    np.testing.assert_allclose(pred, 5.)
    np.testing.assert_allclose(chosen, .1)


def test_bounds_are_validated_per_target():
    with pytest.raises(ValueError, match='prediction bounds'):
        empirical._bound_prediction(np.ones((2, 2)), (np.array([1.]), np.array([5.])))
