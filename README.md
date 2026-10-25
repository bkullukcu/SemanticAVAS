# SemanticAVAS

**Analysis of perceptual ratings and selection of evaluated vehicle warning sounds.**

SemanticAVAS organizes listener ratings by participant, sound and semantic scale. It builds stimulus profiles, compares them with a specified target and ranks sounds already assessed by listeners. Separate tools produce digital sound prototypes and support analysis of acoustic features. A ranking is not a prediction of how listeners will rate an unheard sound.

Current version: `v0.8.2` (research-stage API).

## Install

Use Python 3.10 or newer:

```bash
python -m pip install -e .
python examples/quickstart.py
```

The example creates synthetic ratings and checks the ranking and synthesis routines. To run the test suite:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

## Import listener ratings

The converter accepts a CSV rating table or an NPZ archive. Scale names, limits and direction are declared in a JSON schema rather than inferred from column labels. The sample below contains synthetic ratings, not participant data from our studies.

```bash
semanticavas convert examples/step3_user_data/ratings_wide.csv \
  --schema examples/step3_user_data/schema.json \
  --output-dir example_import

semanticavas rank example_import/ratings.csv \
  --target '{"truck_like":70,"annoying":20}' \
  --scale-ranges '{"truck_like":100,"annoying":100}' \
  --output example_ranking.csv
```

The import creates a normalized ratings file, a coverage table and a conversion manifest. Target values in the example are illustrative. For real ratings, choose targets, scale widths and weights from the actual questionnaire and research question.

See [the input guide](docs/USER_DATA_IMPORT.md) for long-format tables, repeated trials, missing ratings and NumPy arrays.

## Interpretation

Semantic scales are study-specific. Scores from a 1–5 soundscape questionnaire cannot be treated as if they were 0–100 AVAS ratings. Participant and stimulus identifiers must be preserved to avoid mixing independent and repeated observations. Prototype waveforms and acoustic similarity do not demonstrate perceptual acceptance or regulatory compliance.

No empirical ratings, audio, or third-party derived tables are included in this repository. The tests and quickstart use synthetic fixtures. See [research notes](docs/RESEARCH_NOTES.md) for the source-study context and limitations.

## Repository contents

- `src/semanticavas/` — semantic analysis, synthesis, validation and import tools
- `examples/` — synthetic quickstart and example CSV/NPZ inputs
- `tests/` — unit and regression tests
- `docs/` — import reference and research notes

## Citation and license

See [CITATION.cff](CITATION.cff) for software citation information. Experimental datasets should be cited separately. Code is supplied with an MIT license; institutional authorization must be confirmed before public release. The framework does not contain the original participant-level datasets.
