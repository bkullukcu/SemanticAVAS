# Training-domain warnings, listening batches and calibration

This module supports **research study design** independently of the
released WAV model. It never introduces fabricated participant ratings.

## Distribution shift screening

`fit_feature_domain(X_training, groups=source_groups)` standardizes using
only training sounds. Its heuristic threshold is the 95th percentile of
nearest-sound distances when whole training source groups are withheld.
`reference.assess(X_new)` returns nearest distance and a warning flag.
The flag is **not** a probabilistic confidence statement. Compare
performance on a separate unseen source before adopting thresholds.

## Active listening selection

`propose_listening_batch(candidate_features, ids, known_features,
n_select=..., candidate_groups=..., reserved_groups=...,
estimated_uncertainty=...)` uses greedy feature-space novelty and optional
externally estimated uncertainty. Reserved validation groups cannot be
selected as training sounds. Use source quotas to improve diversity.
A new listening test should collect actual participant responses and
maintain the original questionnaire and presentation-level calibration.

## Prediction intervals

`group_conformal_radius(y_calibration, y_pred_calibration,
source_group_labels, alpha=.10)` computes a per-scale groupwise maximum
absolute residual quantile. Predictions are bounded by ± this radius
only after training **and model selection** have finished. At 90%
nominal coverage, six independent sources are insufficient to produce
a finite group-conformal bound. Even with more groups, transfer assumes
exchangeable independent source-group calibration data.

## Listener-aware experimental weighting

`fit_listener_weighted_ridge` takes per-sound SEM estimates calculated
**only from the training listeners**. `evaluate_listener_weighting`
performs source-group-nested tuning and outer evaluation against a training
mean. Regularized inverse-SEM weighting reduces the leverage of noisy
stimulus means but cannot guarantee a predictive improvement. This model
has not been calibrated or evaluated on the restricted 35-WAV collection;
keep it separate from the released predictor.

## Weighting assumptions

`sample_weight_from_sem` transforms measured SEM estimates into bounded
inverse-variance weights. These weights must be calculated from the
training data only during CV; they have not been shown to improve
performance. Distinguish listener disagreement, measurement uncertainty,
and out-of-domain model error.
