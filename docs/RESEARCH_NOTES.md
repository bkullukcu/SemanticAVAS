# Research notes

The example ratings and NumPy files are generated for reproducible demonstrations. Original AVAS, ARAUSv2 and EVAT responses and audio are maintained with their respective datasets. Use their published questionnaires and train/test partitions when reproducing experimental results.

Keep rating constructs and scale ranges tied to their original questionnaires: ARAUSv2 uses 1–5 soundscape judgments, while AVAS semantic ratings may use 0–100 scales. The standard target-ranking workflow compares previously rated sounds. Generated waveforms can be taken forward to calibrated listening studies and regulatory assessment where relevant.

To generate a numerical demonstration, run:

```bash
python examples/reproduce.py --output example_output
```

The generated targets provide an explicit numerical reference for the ranking example.

## Acoustic-to-semantic prediction

The prediction module uses ten digital descriptors and a multi-output ridge model. Its validation protocol separates acoustic stimuli across the outer folds and chooses regularization within each training fold. The reported mean absolute errors are tied to the supplied questionnaire, recordings and held-out grouping. Family identifiers should reflect known recording or synthesis relationships.
