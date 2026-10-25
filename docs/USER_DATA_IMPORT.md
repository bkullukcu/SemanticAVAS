# Importing listener ratings

Use independently collected participant-level ratings without conflating semantic scales from different studies. Raw WAVs, sound-pressure calibration, and vehicle-class legality are **not** inferred from subjective labels.

## Quick start

```bash
python -m pip install -e .
semanticavas convert examples/step3_user_data/ratings_wide.csv \
  --schema examples/step3_user_data/schema.json \
  --output-dir my_import
semanticavas rank my_import/ratings.csv \
  --target '{"truck_like":70,"annoying":20}' \
  --scale-ranges '{"truck_like":100,"annoying":100}' \
  --output ranked.csv
```

First command writes `ratings.csv` and `conversion_manifest.json` with SHA-256 hashes, scale declarations, counts and missing-data policy, plus `coverage_by_stimulus.csv` to reveal incomplete repeated-measures designs. The second uses the existing SemanticAtlas API: it ranks **measured** sounds, not synthetic WAVs that listeners have never rated. Rankings must use legitimate scale widths, explicit target scale directions and meaningful weights. The shown target is purely illustrative.

## JSON schema

`layout` is `wide` or `long`. All studies declare `scales` explicitly, including `min`, `max`, `direction` (`higher_is_more` or `higher_is_less`). The column mapping contains the participant and stimulus IDs. For wide input each scale maps to one numeric column. For long input map `scale` and `rating`; a `(participant,stimulus[,trial],scale)` cell must be unique. The converter preserves the original scale direction; it does **not** reverse-scale a high score automatically.

Repeated evaluations are only allowed with `allow_repeated_ratings: true` and an explicit `trial` column. `missing_policy` defaults to `error`; an opt-in `drop_incomplete` drops entire incomplete ratings rows and discloses the number removed. Original `SemanticAtlas` intentionally averages repeated responses within participant-stimulus before giving participants equal weight.

Optional `metadata_csv` relative to the JSON file joins exclusively by an explicitly unique stimulus ID, never spreadsheet row position; optionally set `metadata_stimulus_column`. Acoustic calibration, stimulus origin and measurement conditions should be recorded in the metadata **only if actually known**. `stimulus_metadata.csv` is exported separately.

## NumPy input

NPZ keys: `ratings` (N,S), `participants` (N), `stimuli` (N), `scales` (S strings), optional `trials` (N). The `scales` strings must exactly match the JSON schema; arbitrary scale order is allowed as long as array columns match those strings. Object/pickle arrays are rejected.

```python
from semanticavas.user_data import convert
report = convert('my_ratings.npz', 'schema.json', 'analysis_input')
from semanticavas import SemanticAtlas
import pandas as pd
atlas = SemanticAtlas(pd.read_csv('analysis_input/ratings.csv'), ['truck_like','annoying'])
```

## Scientific interpretation

Subjective ratings from different protocols are not interchangeable by default (e.g., 1–5 ARAUS soundscape judgments versus 0–100 truck-specific semantic scores). Training/test group splits must be defined by the actual study design; simply parsing IDs does not validate independence. Human listening protocols, audio calibration and approval remain the researcher's responsibility.

A runnable synthetic NumPy example is included:

```bash
semanticavas convert examples/step3_user_data/ratings_numpy.npz --schema examples/step3_user_data/schema.json --output-dir converted_numpy
```
