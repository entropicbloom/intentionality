"""Configuration for the MICrONS representational-ambiguity study."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
CACHE = DATA / "cache"

SEED = 0

# --- quality gates on content labels (not on substrates) -------------------
GOSI_MIN = 0.25        # in-vivo global OSI threshold for orientation labels
CC_ABS_MIN = 0.20      # digital-twin test-set correlation for RF-centre labels

# --- class-level geometric matching ---------------------------------------
N_SPLITS = 200         # random stratified half-splits (reference / test)
K_ORI = 8              # orientation bins (22.5 deg each)
RF_GRID = (3, 3)       # quantile grid for receptive-field bins -> K = 9
K_MAX_EXHAUSTIVE = 9   # 9! = 362,880 permutations, still exhaustive

# --- set-transformer decoder ----------------------------------------------
DEC_POP = 48           # neurons per sampled population (tokens per sample)
DEC_TRAIN_SAMPLES = 24_000
DEC_VAL_SAMPLES = 8_000
DEC_EPOCHS = 10
DEC_BATCH = 128
DEC_DIM = 128
DEC_HEADS = 4
DEC_LAYERS = 2
DEC_LR = 1e-3
DEC_SEEDS = 3
