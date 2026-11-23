# Model promotion gate

Candidate models remain experimental until they outperform the current
predictor on an **independent reserved collection** from at least ten
source groups.

\`assess_model_promotion(observed, candidate, incumbent, source_groups,
independent_test_certified=True)\` compares the two predictors on
exactly the same held-out sounds. It uses a paired bootstrap of entire
source groups, equal-source-average MAE, and a per-scale degradation
guard. Passing a numerical gate only makes the model eligible for
author assessment; no publication or main-branch promotion is automatic.

Mark \`independent_test_certified=True\` only when the study was reserved
before exploring models, acoustic features or hyperparameters. Because
the 35-WAV study has six archival source groups and has already been
used to compare many models, **it cannot pass this gate**. Results of
E0–E4 are development results until additional sources are collected.

This gate is an experiment-management safeguard, not a formal statistical
guarantee with a small number of or nonexchangeable source groups.
