# Importing listener ratings

SemanticAVAS reads participant-level ratings and their questionnaire definitions. The importer retains original semantic scales and exports a normalized table for `SemanticAtlas`.

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

Conversion writes `ratings.csv`, `coverage_by_stimulus.csv` and `conversion_manifest.json`. The manifest records scale definitions, row counts, missing-data handling and SHA-256 hashes. The ranking command compares the sounds represented in the ratings table with the declared target. The target shown above is an example.

## JSON schema

Set `layout` to `wide` or `long`. Define each scale with a `min`, `max` and `direction` (`higher_is_more` or `higher_is_less`). Map participant and stimulus identifiers in `columns`. For wide data, each scale has its own numeric column; for long data, map the `scale` and `rating` columns.

To import repeated ratings, set `allow_repeated_ratings: true` and identify a `trial` column. The analysis averages repeated trials within participant–stimulus pairs before calculating participant-balanced profiles.

The default `missing_policy` is `error`. Select `drop_incomplete` to remove incomplete rows while recording the number removed in the manifest.

An optional `metadata_csv`, relative to the JSON schema, joins on a unique stimulus identifier. Set `metadata_stimulus_column` when that identifier has a different name in the metadata file. Include available stimulus origin, recording conditions and calibration information. Imported stimulus metadata are exported as `stimulus_metadata.csv`.

## NumPy input

An NPZ input stores `ratings` (N, S), `participants` (N), `stimuli` (N), `scales` (S strings), and optionally `trials` (N). Array-column order follows the `scales` labels, which must agree with the JSON schema. Use numeric and string arrays.

```python
from semanticavas.user_data import convert
from semanticavas import SemanticAtlas
import pandas as pd

report = convert('my_ratings.npz', 'schema.json', 'analysis_input')
atlas = SemanticAtlas(
    pd.read_csv('analysis_input/ratings.csv'),
    ['truck_like', 'annoying'],
)
```

The included NumPy example is ready to run:

```bash
semanticavas convert examples/step3_user_data/ratings_numpy.npz \
  --schema examples/step3_user_data/schema.json \
  --output-dir converted_numpy
```

## Measurement context

Preserve the questionnaire meaning, scale range and response direction in study metadata. For example, ARAUSv2 soundscape ratings use 1–5 scales, while vehicle-specific semantic experiments may use 0–100 ratings. Define participant and stimulus group splits from the experimental protocol, and record listening conditions and calibration alongside results.
