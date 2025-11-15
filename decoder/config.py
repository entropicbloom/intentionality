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

    Raises:
        FileNotFoundError: If the underlying directory doesn't exist
    """
    if not UNDERLYING_DIR.exists():
        raise FileNotFoundError(
            f"Underlying directory not found: {UNDERLYING_DIR}\n"
            f"Expected at: {PROJECT_ROOT / 'underlying'}\n"
            f"Please ensure the 'underlying' directory exists in the project root."
        )
    return str(UNDERLYING_DIR / relative_path)


def validate_config(config, required_keys=None):
    """
    Validate configuration dictionary and provide helpful error messages.

    Args:
        config: Configuration dictionary to validate
        required_keys: Optional list of required keys to check

    Returns:
        bool: True if validation passes

    Raises:
        ValueError: If required keys are missing or values are invalid
    """
    if required_keys is None:
        required_keys = ['model_class_str', 'dataset_class_str']

    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(
            f"Missing required config keys: {missing_keys}\n"
            f"Current config keys: {list(config.keys())}"
        )

    # Validate hidden_dim structure
    hidden_dim = config.get('hidden_dim', [50, 50])
    if not isinstance(hidden_dim, list) or len(hidden_dim) == 0:
        raise ValueError(
            f"'hidden_dim' must be a non-empty list, got: {hidden_dim}"
        )

    return True

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