# SoftwareX peer-review revision: methods and limitations

This document accompanies manuscript SOFTX-D-26-01551. The original
v0.9.0 release remains available at DOI 10.5281/zenodo.23661611.
The revision branch is **not** a new archived software release.

## Two distinct operations

1. Catalogue ranking requires listener ratings for every sound in the
   candidate catalogue. It cannot evaluate an unrated waveform.
2. WAV prediction requires WAV/rating pairs for model training and may
   estimate scores for further recordings. It is not a validated replacement
   for listening tests across independent recording sources.

The stationary harmonic/noise synthesizer and parameter-domain
`RidgeSurrogate` are experimental utilities. The ARAUS example
uses the original ARAUS descriptors and folds; it does not evaluate
the ten-descriptor AVAS WAV pipeline.

## Input and scale contracts

`SemanticAtlas.rank` delegates to `rank_existing_sounds`. The default
scale width is 100, suitable for 0--100 questionnaires; other scales
require explicit widths. Participant means are calculated after repeated
trials are averaged.

The WAV model's rating range defaults to 0--100 and can be overridden:

```bash
semanticavas train-predictor \
  --manifest manifest.csv --ratings ratings.csv \
  --scales pleasant annoying --scale-bounds '{"pleasant":[1,5],"annoying":[1,5]}' \
  --missing-policy stimulus_mean --full-recording \
  --output-dir fitted
```

The default missing policy is `error`. The opt-in
`stimulus_mean` method fills missing listener/sound/scale cells only
after repeat averaging, using the available stimulus mean. It preserves
that stimulus's mean but underestimates within-sound response variability.
Never treat completed cells as independent observations in uncertainty
estimation. The imported catalogue has a separate missingness policy.

## Audio front-end

- Original WAV sampling rate, no resampling (minimum 8 kHz).
- Integer PCM converted to signed full-scale digital amplitudes.
- Analysis of the first 5 seconds by default, or the complete WAV with
  `--full-recording` / `max_seconds=None`.
- Independent channel descriptors averaged at the end (no waveform
  summation or cancellation).
- Welch PSD: 4,096-sample Hann windows, 50% overlap, constant detrend,
  density scaling. Frequency region from 20 Hz to min(10 kHz, Nyquist).
- Energy bands: below 500 Hz, 500--2,000 Hz, and at least 2,000 Hz
  inside the frequency region.
- RMS is in digital dBFS, not SPL; zero crossings are a per-sample
  fraction, not crossings per second.

No playback gain calibration, time-varying descriptors, perceptual
loudness, sharpness, roughness, fluctuation strength or modulation
descriptors are implemented in the present ten-feature front end.

## Re-analysis on the restricted original study data

Thirty-five verified WAVs, 20 listeners, seven 0--100 scales and 4,896
observed rating cells were analysed without publishing the recordings,
participants or feature-level identifiers. All model selection and
standardization was confined to each outer-training split.

The following aggregate mean absolute errors were obtained:

| Outer holdout | Training mean | Ridge | RBF SVR | Random forest |
| --- | ---: | ---: | ---: | ---: |
| One sound | 13.58 | 13.41 | 12.77 | 11.40 |
| One of six archival source blocks | 15.41 | 16.39 | 14.80 | 17.72 |

For paired MAE difference against the training mean, a 10,000-replicate
percentile bootstrap over held-out sounds gives Ridge [-0.28, -0.08],
SVR [-1.40, -0.21], random forest [-4.27, -0.29].
Resampling six held-out source blocks gives Ridge [+0.11, +1.54],
SVR [-3.87, +1.10], random forest [-0.24, +3.53].
These are descriptive intervals; six blocks do not establish robust
transfer to unseen source families.

With nested ridge fitting and otherwise unchanged WAV features:

| Extraction | One-sound MAE | Source-block MAE |
| --- | ---: | ---: |
| 2 s | 13.40 | 16.13 |
| 5 s | 13.41 | 16.39 |
| 10 s | 13.28 | 16.65 |
| Full WAV | 13.28 | 16.65 |

The original WAVs last 1.92 to 9.90 seconds, so 10-second and full
recording features coincide. Mean listener-rating standard error is
5.10 points. These empirical findings need the consent-restricted audio
and ratings for full independent numerical reproduction; synthetic CI
fixtures and tutorial waveforms do not reproduce them.

## Scientific scope

The screening checks of at least ten rated sounds and two listeners per
sound are algorithmic safeguards, not sample-size recommendations.
The synthetic tutorial creates synthetic ratings and only demonstrates
the workflow. The study's source groups were defined by archival
provenance, not independently verified synthesis or manufacturer
identity. New design families need further human validation.

## Testing

Run `python -m pytest -q` in an editable installation. Regression tests
cover matching ranking losses, full-WAV extraction and explicit
rating-scale and missing-cell contracts. Verify CI on the actual PR head;
tests on a local snapshot are not a substitute for CI.
