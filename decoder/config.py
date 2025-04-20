# Configuration dictionary for decoder experiments

# Directory for saved underlying models
MODELS_DIR = 'saved_models/'

# Default Configuration
config = {
    "model_class_str": 'fully_connected_dropout',
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