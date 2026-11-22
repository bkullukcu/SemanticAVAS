# SemanticAVAS — development roadmap

The items below guide future development and do not assert scientific findings.

## P0: Traceability and data integrity
- Keep package version, citation metadata and README DOI synchronized with CI tests.
- Add regression coverage for repeated ratings, listener IDs, incomplete panels, invalid scale ranges and polarity.
- Test exact stimulus/WAV/family mapping and reject missing or ambiguous identifiers.
- Keep separate workflows for ranking evaluated sounds and prediction on unheard WAVs.

## P1: Statistical robustness
- Validate listener-cluster bootstrap against deterministic fixtures; document interval interpretation.
- Add nested stimulus/family holdout checks and training-only acoustic-feature preprocessing tests.
- Compare sound-level prediction to training-mean baselines without leaking held-out stimuli.

## P2: Experiment interoperability
- Export stimulus manifests, questionnaire definitions, data hashes and evaluation decisions.
- Clearly distinguish synthetic demonstrations, available external datasets and actual listening experiments.
- Document model application-domain boundaries and calibration requirements.

## Acceptance criteria
Each change must show its test or documented inspection result; experimental participant ratings, calibrated levels, model performance and UNECE compliance must never be invented.
