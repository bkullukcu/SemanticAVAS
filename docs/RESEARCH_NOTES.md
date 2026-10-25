# Research notes

The example ratings and NumPy files are synthetic. The package does not redistribute the original listener responses, AVAS stimulus audio, ARAUSv2 recordings or EVAT data. Researchers must obtain external datasets under their own terms and preserve the original questionnaire definitions and train/test splits.

Ratings from different studies are not exchangeable without an explicit measurement model. In particular, ARAUSv2 1–5 ratings should not be interpreted as equivalent to AVAS 0–100 semantic judgments. A semantic target ranking refers to sounds that were actually rated; it does not validate human judgments of new synthetic warning sounds. Digital prototypes are not calibrated regulatory demonstrations.

To generate a numerical demonstration, run:

```bash
python examples/reproduce.py --output example_output
```

The generated targets are synthetic and do not stand in for perceptual observations.
