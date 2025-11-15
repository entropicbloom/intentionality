# Configuration dictionary for decoder experiments
import os
from pathlib import Path

# Get the project root directory (intentionality/)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Directory for saved underlying models (relative to project root)
UNDERLYING_DIR = PROJECT_ROOT / 'underlying'
MODELS_DIR = 'saved_models/'

def get_underlying_path(relative_path):
    """
    Get absolute path to a file/directory in the underlying directory.

    Args:
        relative_path: Path relative to the underlying directory

    Returns:
        Absolute path as a string
    """
    return str(UNDERLYING_DIR / relative_path)

# Default Configuration
config = {
    "model_class_str": 'fully_connected',
    "dataset_class_str": 'mnist',
    "decoder_class": 'TransformerDecoder',
    "preprocessing": 'multiply_transpose',
    "untrained": False,
    "varying_dim": False,
    "hidden_dim": [50, 50],
    "num_neurons": 10,
    "min_neurons": 2,
    "use_target_similarity_only": False,
    "models_dir": MODELS_DIR  # Include models directory in config
} 