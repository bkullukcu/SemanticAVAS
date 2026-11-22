# AGENTS.md — SemanticAVAS

Participant-balanced semantic analysis and WAV-to-semantic prediction of vehicle warning sounds.

Read `README.md`, `ROADMAP.md`, `docs/RESEARCH_NOTES.md`, `docs/AI_DISCLOSURE.md`, `pyproject.toml` and relevant source/tests before editing.

## Autonomous development rules
1. Check open issues, PRs and `main` before choosing one narrowly scoped, verifiable task. Do not duplicate an active PR.
2. Branch only from `ai-dev`, change code and focused tests in a feature branch, and create a pull request with base `ai-dev`. Never push directly to `main` or `ai-dev`; never open or merge a PR targeting `main`, enable repository-wide auto-merge, or publish a release. Automated squash-merge into `ai-dev` alone is authorized if every gate below passes.
3. Prefer reproducible numerical tests, explicit input contracts, and stable user-facing APIs. Justify backward-incompatible changes before proposing them.
4. Never fabricate measurements, listeners, uncertainty estimates, acceptance thresholds, benchmark outcomes, citations or DOI records.
5. Identify AI-assisted changes in the PR and document assumptions and limitations. Keep examples and synthetic fixtures distinguishable from empirical data.
6. Preserve original data rights: no confidential, restricted or third-party measurements in public commits.

## Required verification
```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python examples/quickstart.py
python -m build
python -m twine check dist/*
```

Run available checks and report only observed results. GitHub Actions validates the supported Python/OS matrix. If CI fails, diagnose the code rather than relaxing the tests. Use `tests/test_public_contract.py` to protect release metadata and syntax.

## Scientific invariants
- Participants are bootstrap/resampling clusters. Average repetitions within participant and stimulus first.
- Keep questionnaire constructs, scale polarity and rating ranges explicit; do not silently mix 1–5 and 0–100 scales.
- Train/tune within splits and hold out independent stimuli or sound families to prevent acoustic leakage.
- Incomplete listener–stimulus panels cannot be treated as complete repeated-measures samples.
- Synthesized waveforms are uncalibrated demonstrations, not validated regulatory AVAS signals.
- Prediction metrics from synthetic data are not human-listener findings; avoid out-of-domain generalization claims.
## PR acceptance checklist
- Provide problem statement and reproducible failure or clear docs/metadata rationale.
- Add focused regression tests for substantive changes.
- Report tests executed, missing checks, and CI state, including any unverified external datasets.
- Do not silently modify citation, license, file schemas, algorithms, audio calibration claims, or scientific metrics.
- Automated **squash merge into `ai-dev` only** is allowed when all six CI jobs on the current PR head have completed successfully, the PR is mergeable, no unresolved PR discussions or blocking remarks exist, and the change is narrowly scoped with no altered CI/workflow, package release, access control, science claims, or incompatible public behavior. Check that the expected head SHA has not changed immediately before merging. Skip/hold the PR if any check is missing or ambiguous.
- Keep `main` stable: any promotion from `ai-dev` to `main` requires explicit human approval. Never silently retarget a PR to `main`.
- Create a dated `ai-snapshots/YYYY-MM-DD` branch from the `ai-dev` head on weekly audit days, unless it already exists. A snapshot is a reference for recovery; do not force-update or overwrite it.
- Exclude release/version/DOI changes, scientific interpretation, new unvalidated algorithms, security/workflow changes, and schema-breaking changes from unattended merging; leave those PRs in draft and report them for human assessment.
