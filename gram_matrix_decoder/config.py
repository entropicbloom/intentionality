"""Configuration for gram matrix decoder experiments."""

from pathlib import Path

# Reference model configuration
REFERENCE_MODEL_TYPE = "fully_connected_dropout"
REFERENCE_DATASET_TYPE = "mnist"
REFERENCE_HIDDEN_DIM = "[100]"
REFERENCE_UNTRAINED = False

# Evaluation model configuration
EVAL_MODEL_TYPE = "fully_connected_dropout"
EVAL_DATASET_TYPE = "mnist"
EVAL_HIDDEN_DIM = "[50,50]"
EVAL_UNTRAINED = False

REFERENCE_SEEDS = range(0, 5)
TEST_SEEDS = range(10, 15)
N_RANDOM_PERMS = 100_000
ALL_PERMS = True
SAVE_DISTANCES = True

BASEDIR = Path("../underlying/saved_models")
REFERENCE_UNTRAINED_SUFFIX = "-untrained" if REFERENCE_UNTRAINED else ""
EVAL_UNTRAINED_SUFFIX = "-untrained" if EVAL_UNTRAINED else ""
REFERENCE_MODEL_FMT = f"{REFERENCE_MODEL_TYPE}-{REFERENCE_DATASET_TYPE}{REFERENCE_UNTRAINED_SUFFIX}-hidden_dim_{REFERENCE_HIDDEN_DIM}/seed-{{seed}}"
EVAL_MODEL_FMT = f"{EVAL_MODEL_TYPE}-{EVAL_DATASET_TYPE}{EVAL_UNTRAINED_SUFFIX}-hidden_dim_{EVAL_HIDDEN_DIM}/seed-{{seed}}"

TOLERANCE = 1e-13
RANDOM_SEED = 42