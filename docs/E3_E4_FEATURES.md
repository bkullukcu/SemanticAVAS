# Calibrated psychoacoustic E3 and frozen-CLAP E4

Both are optional, isolated research adapters. Neither changes the public
ten-feature predictor or makes an empirical performance claim.

## E3 — calibration first

The required quantity \`pa_per_fullscale\` is the **verified physical
pressure in Pa for unit digital amplitude** of each recording. Without
a known calibration chain, do not run E3 or pretend the WAV is in SPL.
With the \`mosqito\` optional dependency installed, call
\`calibrated_mosqito_features(wav, pa_per_fullscale=...)\`.
It computes ISO 532-1 stationary loudness, DIN sharpness,
Daniel–Weber roughness and ECMA tonal tone-to-noise ratio.
Signals are explicitly resampled to 48 kHz. The first 5 seconds are
an optional configurable window.

## E4 — CLAP latent audio features

Install \`laion-clap\` separately because its torch/numpy requirements
may conflict with the standard SemanticAVAS environment.
Download and pin an approved pretrained checkpoint outside the Git repo.
Use \`fixed_clap_embeddings(paths, stimulus_ids, checkpoint_path=...)\`
to generate an embedding CSV and a checkpoint SHA256 record. The adapter
**never downloads a model automatically**. Each WAV is fingerprinted.
On import, \`checked_embedding_matrix\` verifies stimulus IDs and file
SHA256 against the current manifest. Retain these tables privately.

LAION-CLAP API: https://github.com/LAION-AI/CLAP
MoSQITo metric APIs: https://mosqito.readthedocs.io/

## Common training/evaluation

Use \`nested_representation_test({"E0": x0, "E3": x3, "E4": x4}, y,
source_groups)\` with matrices aligned to the same stimulus order.
Standardization, optional PCA and Ridge hyperparameter tuning are
fitted within each training fold. Entire recording source groups remain
held out in both outer and inner loops.

Keep the original E0–E2 experimental matrix as a reference. E3 must
wait for physical calibration; E4 must wait for local checkpoint and
model initialization. Current 35-WAV results do not establish a gain
for either method. Record source groups and reserve new sources for
confirmatory testing.
