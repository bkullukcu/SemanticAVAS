# E0–E2 acoustic feature development study

This is an **exploratory evaluation**, independent of the released SemanticAVAS
v0.9.0/v0.9.1 WAV predictor. It adds no weights or trained models to the
public repository and does not amend published numerical results.

## Hypotheses and prespecified comparisons

- **E0:** the original ten digital descriptors from `acoustic_features.py`.
- **E1:** E0 + maximum spectral-peak excess over a nearby noise-floor
  median (dB) + fraction of 80 Hz–10 kHz power within ±25 Hz of the
  dominant spectral peak. These are tonal *proxies*, not standardized tonality.
- **E2:** E0 + coefficient of variation of 50-ms frame RMS + mean positive
  spectral flux from normalized 64-ms magnitude spectra with 50% overlap.
  These are digital signal dynamics, not psychoacoustic fluctuation strength.

All three settings use the same audio window and source/scale definitions.
No model can see semantic ratings of an outer-held-out sound or family
when computing scaling or hyperparameters.

## Reproduce locally with authorized data

Install optional model dependencies (scikit-learn is named `scikit-learn`
in `pip`, imported as `sklearn` in Python):

```bash
python -m pip install -e ".[dev]"
python -m semanticavas.feature_ablation \
  --manifest /private/manifest.csv \
  --ratings /private/ratings.csv \
  --scales truck_like annoying electric_like buzzing harsh whistling futuristic \
  --output-dir /private/ablation_results
```

The manifest has `stimulus,wav,sound_family`. `sound_family` must be
grounded in archival provenance, experiment-design family, or an actual
recording source; a made-up family definition invalidates the blocked
evaluation. If sound_family is absent, the runner produces only
leave-one-sound-out results, with no cross-source claim.

The ratings file has `participant,stimulus,<scales>`. Repeat presentations
are averaged within participant/stimulus by the original validated loader.
The runner follows the loader's missing-cell default (`error`), so
originally imputed and unmodified datasets must not be silently mixed.

## Model comparison

Ridge (α = 0.1, 1, 10, 100, 1000), RBF-SVR (C = 0.1, 1, 10) and PLS
(n_components = 1, 2, 3) are tuned in grouped **inner** folds. Their
standardization is fitted on each training fold. Each outer holdout
uses a training-mean comparator and optional explicit response bounds.
Score summaries report per-scale, sound-weighted and source-balanced
MAE; source-balanced MAE weights each held-out family equally.

For 35 sounds with only six archival source blocks, uncertainty about
transfer remains substantial. Even if E1/E2 improve these held-out
results, new recording sources should be reserved for independent
confirmation before claiming a superior model.

Outputs under `--output-dir`:

- `feature_provenance.csv`: stimulus ID, source labels, WAV SHA256 and
  four derived features.
- `outer_predictions.csv`: per-held-out-sound predictions and baselines.
- `comparison.csv`: E0/E1/E2 × model × holdout summaries.
- `protocol.json`: feature definitions and configuration.

These outputs can contain restricted identifiers and must remain outside
the public repository. The code and tests generate no measured results.
All signal values are uncalibrated and cannot be interpreted as SPL.
