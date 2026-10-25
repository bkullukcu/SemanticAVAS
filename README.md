# SemanticAVAS

**Listener-based semantic profiling and target matching for vehicle warning sounds.**

SemanticAVAS groups perceptual ratings by listener and evaluated sound, builds profiles on named semantic scales, and compares those profiles with a user-defined target. The package also includes numerical sound-prototype synthesis and tools for controlled model evaluation. It ranks sounds **that have been rated**; it does not predict the perception or safety performance of an untested sound.

**Research software preview · v0.8.2 · Python 3.10–3.13 · MIT**

## Example: from ratings to a sound choice

![Synthetic semantic stimulus profiles with listener-bootstrap intervals](docs/figures/semantic_profiles.svg)

*Illustrative semantic profiles from 18 simulated listeners. Dots are participant-balanced means and whiskers are pointwise bootstrap intervals. No human participant data are included.*

![Synthetic semantic target ranking](docs/figures/target_ranking.svg)

*A two-scale target is compared with four evaluated synthetic sounds; the ranking uses weighted squared distance after explicit scale normalization. This plot demonstrates the decision rule, not preference prediction for a new AVAS design.*

These figures come from [`SemanticAtlas`](src/semanticavas/core.py), not manually assigned ranking results. Regenerate them with `python examples/make_readme_figures.py`.

## Install and check the package

From the repository root:

```bash
python -m pip install -e .
python examples/quickstart.py
```

The example assembles a small synthetic rating table, ranks its sounds and generates a digital waveform. To run the tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

## Work with your own ratings

CSV and NumPy input are supported. A JSON schema declares the rating scales, their ranges and direction, and how participant and stimulus identifiers map to columns. No scale direction or conversion between studies is inferred.

```bash
semanticavas convert examples/step3_user_data/ratings_wide.csv \
  --schema examples/step3_user_data/schema.json \
  --output-dir example_import

semanticavas rank example_import/ratings.csv \
  --target '{"truck_like":70,"annoying":20}' \
  --scale-ranges '{"truck_like":100,"annoying":100}' \
  --output example_ranking.csv
```

The first command creates a normalized ratings table, a coverage report and a conversion manifest. The second compares the already-evaluated sounds with an illustrative target. [Data import](docs/USER_DATA_IMPORT.md) explains wide and long tables, repeated trials, missing ratings and NPZ inputs.

## The target-matching rule

For stimulus $i$ and semantic dimension $k$, `SemanticAtlas.rank` uses

$$
L_i=\frac{\displaystyle\sum_k w_k\left(\frac{\bar{s}_{ik}-t_k}{r_k}\right)^2}{\displaystyle\sum_k w_k},
$$

where $\bar{s}_{ik}$ is the participant-balanced mean rating, $t_k$ the stated target, $r_k$ an externally specified scale width and $w_k$ a nonnegative weight. Smaller $L_i$ means a closer match **within the evaluated catalogue**. Bootstrap resampling is performed at listener level where the design permits it.

The original scale boundaries and questionnaire meanings matter. A soundscape score on a 1–5 scale must not be silently treated as a 0–100 AVAS rating.

## What is included

- Aggregation of listener ratings and uncertainty summaries.
- Ranking of previously rated sounds and sensitivity to target weights.
- Digital harmonic/noise prototype generation and optional data-analysis adapters.
- CSV/NPZ import with explicit scale and identifier checks.

The public examples are wholly synthetic. Neither original listener ratings nor AVAS audio recordings are redistributed. Synthesized prototypes have not been perceptually validated or shown to comply with a warning-sound regulation. See [Research notes](docs/RESEARCH_NOTES.md) and [FAQ](docs/FAQ.md).

## Reproduce the figures

```bash
python examples/make_readme_figures.py
# Optional LaTeX typesetting, if TeX is installed:
python examples/make_readme_figures.py --latex
```

The SVGs under `docs/figures/` are generated from the same data and public functions as the script. No external data download is required.

## Citation and license

Citation metadata: [`CITATION.cff`](CITATION.cff). Source license: [`LICENSE`](LICENSE). Please cite any experimental dataset independently. This software is a research preview and its interface may change before version 1.0.
