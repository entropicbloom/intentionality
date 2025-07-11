"""Configuration for gram matrix decoder experiments."""

from pathlib import Path

MODEL_TYPE = "fully_connected_dropout"
DATASET_TYPE = "mnist"
HIDDEN_DIM = "[50,50]"
UNTRAINED = False

REFERENCE_SEEDS = range(0, 1)
TEST_SEEDS = range(10, 15)
N_RANDOM_PERMS = 100_000
ALL_PERMS = False
SAVE_DISTANCES = True

BASEDIR = Path("../underlying/saved_models")
UNTRAINED_SUFFIX = "-untrained" if UNTRAINED else ""
MODEL_FMT = f"{MODEL_TYPE}-{DATASET_TYPE}{UNTRAINED_SUFFIX}-hidden_dim_{HIDDEN_DIM}/seed-{{seed}}"

TOLERANCE = 1e-13
RANDOM_SEED = 42