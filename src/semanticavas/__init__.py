"""SemanticAVAS research software."""
__version__ = "0.8.2"
from .core import *
from .empirical import (validate_semantic_ratings, aggregate_semantics, bootstrap_profiles,
    rank_existing_sounds, bootstrap_target, leave_one_participant_out,
    pca_stimulus_profiles, nested_leave_one_sound_out, nested_leave_one_group_out)
from .acoustic_features import features_from_wav_bytes, FEATURE_NAMES

from .synthesis35 import load_35_stimulus_ratings, extract_35_wav_features, published_design_table, SCALES as SYNTHESIS35_SCALES

from .decision import admissible_catalog, weight_sensitivity
