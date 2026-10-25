# Predicting semantic ratings from WAV files

`semanticavas.prediction` connects the existing ten-feature WAV extractor to a participant-balanced multi-output ridge model. It trains on paired sound recordings and human responses, evaluates unseen recordings, and writes a reusable model archive.

## Input tables

`manifest.csv` contains one row per sound:

```csv
stimulus,wav,sound_family
S01,audio/S01.wav,harmonic_A
S02,audio/S02.wav,harmonic_A
S03,audio/S03.wav,harmonic_B
```

The shown fragment illustrates the columns; training needs at least ten unique rated WAVs. The optional `sound_family` column should identify sounds sharing a source recording or closely related synthesis variants. If used, provide at least three distinct families, with sufficiently many training examples in each holdout split.

`ratings.csv` contains each participant's responses:

```csv
participant,stimulus,truck_like,annoying
P01,S01,75,20
P02,S01,69,25
P01,S02,62,40
P02,S02,67,38
```

Use numeric 0–100 ratings, identical stimulus IDs in both tables and at least two listeners per stimulus. Multiple trials of the same sound by one listener are averaged first, then listeners are given equal weight. Use the semantic scales defined by the original questionnaire and consistent stimulus-level audio material. A WAV is a sound stimulus; listener observations are kept together during validation.

## Run a complete example

```bash
python -m pip install -e ".[dev]"
python examples/prediction_demo.py --output-dir prediction_demo_output
```

This creates 16 deterministic digital WAV prototypes from four families, synthetic participant ratings, a held-out validation table, a fitted model and an unseen WAV prediction. The values demonstrate software behavior and file formats.

## Fit and evaluate a model

```bash
semanticavas train-predictor \
  --manifest my_data/manifest.csv \
  --ratings my_data/ratings.csv \
  --scales truck_like annoying \
  --max-seconds 5 \
  --output-dir trained_model
```

The extractor takes the first `max_seconds` of each PCM WAV and reports ten digital descriptors: RMS in dBFS, spectral centroid, spread, 90% roll-off, fractional low/mid/high band energy, spectral flatness, crest factor and zero-crossing rate. Keep recording and playback gain conventions consistent among the training and prediction recordings.

Training performs nested cross-validation. The outer split holds out each sound, or each declared family. The inner split chooses the ridge penalty without using the outer held-out targets. The predictor is then fitted once to all training sounds using an internally cross-validated penalty. Each regression is standardized on its own training subset.

Output files:

| File | Contents |
| --- | --- |
| `semantic_model.npz` | Pickle-free fitted regression coefficients, feature normalization and metadata |
| `heldout_predictions.csv` | Observed, predicted and baseline rating for each outer held-out sound |
| `validation.json` | Per-scale MAE, baseline MAE, 90th-percentile absolute residual and fit configuration |
| `audio_summary.csv` | Per-sound sampling rate, processed duration, channel count and listener count |

Models are trained from the user's own recordings and ratings. The repository supplies a self-contained synthetic tutorial and the fitting procedure.

## Predict an unseen sound

```bash
semanticavas predict-wav \
  --model trained_model/semantic_model.npz \
  new_sound.wav second_sound.wav \
  --output predictions.csv
```

The CSV gives one row per WAV, with predicted semantic values and the following descriptors:

- `outside_training_feature_ranges`: whether any new descriptor is beyond the extrema of the training sounds.
- `nearest_training_distance_standardized`: Euclidean distance to the nearest training example after training-set standardization.
- `prediction_clipped_to_rating_scale`: whether a regression value was clipped to 0–100.

The command's JSON output also includes audio metadata and the cross-validated 90th-percentile absolute residual for each scale. The error summary describes the validation sample. For assessment of a new acoustic family, prioritize family-blocked validation and inspect the comparison against the baseline.

## Python API

```python
from semanticavas.prediction import train_wav_predictor, save_predictor, load_predictor

model, heldout, audio_summary = train_wav_predictor(
    'my_data/manifest.csv', 'my_data/ratings.csv',
    ['truck_like', 'annoying'],
)
save_predictor(model, 'semantic_model.npz')
output = load_predictor('semantic_model.npz').predict_wav('new_sound.wav')
print(output['truck_like'], output['annoying'])
print(model.validation['cv_mae_by_scale'])
```

The NPZ archive stores model parameters and training feature vectors for nearest-neighbor distance assessment, but omits listener identifiers and audio bytes. Retain the training tables privately when the study includes participant data. Use the same questionnaire constructs and audio processing conventions when applying the model to future recordings.

## Plot held-out predictions

```bash
python examples/prediction_demo.py
python examples/plot_prediction_validation.py --table prediction_demo_output/heldout_predictions.csv
```

The resulting figure is `docs/figures/prediction_validation.svg`. It uses the generated sound families and synthetic listener ratings from the tutorial. Replace the input CSV with your own `heldout_predictions.csv` to inspect real-study errors.
