# Error diagnostics and listener variability

This analysis is separate from model training. It takes only *outer-held-out*
predictions and, optionally, observed participant-level ratings.

Run from a private workspace:

\`\`\`bash
python -m semanticavas.diagnostics \
  --predictions /private/ablation/outer_predictions.csv \
  --ratings /private/ratings.csv \
  --scales truck_like annoying \
  --output-dir /private/diagnostics
\`\`\`

Reports:

- \`models.csv\`: sound-weighted MAE and paired source-balanced bootstrap
  intervals relative to the outer-training mean
- \`scales.csv\`: scale-wise MAE, bias and out-of-sample R²
- \`sources.csv\`: error and baseline for each withheld recording family
- \`listener_variability.csv\`: observed listener disagreement, SEM and IQR,
  after balancing repeated presentations
- \`listener_split_half.csv\`: random split-half correlation across sounds

Intervals are **descriptive** with a small number of source groups. Source
bootstrap replicates entire groups, not their individual recordings. Do
not interpret the intervals as independent validation of a model chosen
using those same groups. These outputs may contain restricted identifiers.

The observed-rating summaries describe measurement variability rather than
predicted-score uncertainty. The latter needs an independent calibration
collection before coverage claims can be made.


## Optional mixed-effects assessment

With \`statsmodels\` installed, add \`--mixed-effects\` to the diagnostics
CLI. The formula includes fixed stimulus effects and a random
participant intercept after repeated-trial balancing. It reports
participant-intercept variance, residual variance and an ICC.
This is an exploratory measure of listener-specific response offsets;
it does not measure predictive accuracy on an unseen recording source.
Examine the reported convergence flag before interpreting estimates.
