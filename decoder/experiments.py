# Import base config and experiment setup functions
from decoder.config import config as base_config
from decoder.setup.class_id import setup_and_train as setup_and_train_class_id, setup_and_train_mixed_hidden_dims
from decoder.setup.input_pixel import setup_and_train as setup_and_train_input_pixel

def run_ablation_experiments_classid(
    min_neurons=None, max_neurons=None, num_seeds=5, experiment_config=None,
    project_name="decoder-neuron-ablation"
):
    """
    Run ablation experiments by varying the number of neurons.

    Args:
        min_neurons (int, optional): Minimum number of neurons to use. Defaults to config value.
        max_neurons (int, optional): Maximum number of neurons to use. Defaults to config value.
        num_seeds (int, optional): Number of random seeds to use. Defaults to 5.
        experiment_config (dict, optional): Configuration to use. Defaults to base config.
        project_name (str, optional): W&B project name.
    """
    # Use base config if no specific config is provided
    if experiment_config is None:
        experiment_config = base_config.copy()

    # Use config values as defaults if not provided
    min_neurons = min_neurons if min_neurons is not None else experiment_config['min_neurons']
    max_neurons = max_neurons if max_neurons is not None else experiment_config['num_neurons']

    # Loop through different numbers of neurons
    for num_neurons in range(min_neurons, max_neurons + 1):
        print(f"Running experiments with {num_neurons} neurons")
        # Create a copy of the config for this specific number of neurons
        current_config = experiment_config.copy()
        # Update the num_neurons in the config
        current_config['num_neurons'] = num_neurons
        for seed in range(num_seeds):
            setup_and_train_class_id(seed, num_neurons, project_name=project_name, config=current_config)

def run_main_experiments_classid(
    num_seeds=5, 
    project_name="decoder-main-experiments"
):
    """
    Run experiments with all neurons for different configurations:
    1. The current config
    2. A config with model_class_str='fully_connected'
    3. A config with untrained=True
    4. A config with varying_dim=True

    Each configuration is run with all neurons for multiple seeds.

    Args:
        num_seeds (int, optional): Number of random seeds to use. Defaults to 5.
        project_name (str, optional): W&B project name.
    """
    # Save the original config
    original_config = base_config.copy()

    # --- Configuration 1: Fully Connected Dropout (current base) ---
    print("Running with model_class_str='fully_connected_dropout'")
    fcb_config = original_config.copy()
    fcb_config["model_class_str"] = 'fully_connected_dropout'
    for seed in range(num_seeds):
        setup_and_train_class_id(seed, fcb_config['num_neurons'], project_name=project_name, config=fcb_config)

    # --- Configuration 2: Fully Connected --- 
    print(f"Running with model_class_str='fully_connected'")
    fc_config = original_config.copy()
    fc_config["model_class_str"] = 'fully_connected'
    for seed in range(num_seeds):
        setup_and_train_class_id(seed, fc_config['num_neurons'], project_name=project_name, config=fc_config)

    # --- Configuration 3: Untrained Fully Connected ---
    print(f"Running with model_class_str='fully_connected', untrained=True")
    untrained_config = original_config.copy()
    untrained_config["untrained"] = True
    untrained_config["model_class_str"] = 'fully_connected'
    for seed in range(num_seeds):
        setup_and_train_class_id(seed, untrained_config['num_neurons'], project_name=project_name, config=untrained_config)

    # --- Configuration 4: Varying Dimension --- 
    # Note: Base config's model_class_str is used here unless overwritten
    print(f"Running with varying_dim=True (model: {original_config['model_class_str']})")
    varying_dim_config = original_config.copy()
    varying_dim_config["varying_dim"] = True
    for seed in range(num_seeds):
        setup_and_train_class_id(seed, varying_dim_config['num_neurons'], project_name=project_name, config=varying_dim_config)

def run_main_experiments_inputpixels(num_seeds=5, project_name="decoder-inputpixels"):
    """
    Run experiments decoding input pixel position for different encoding types.
    Uses the first layer weights and column-wise cosine similarity.

    Args:
        num_seeds (int, optional): Number of random seeds to use. Defaults to 5.
        project_name (str, optional): W&B project name.
    """
    # Base config - modify as needed for these experiments
    base_config_copy = base_config.copy()
    base_config_copy['varying_dim'] = False # Or True, depending on which models to test
    # Set a hidden_dim known to exist for the models being loaded
    base_config_copy['hidden_dim'] = [50, 50] # Example: Adjust if needed
    base_config_copy['decoder_class'] = 'TransformerDecoder' # Or FCDecoder
    # Preprocessing is handled inside FirstLayerDataset, remove from main config?
    if 'preprocessing' in base_config_copy: del base_config_copy['preprocessing']

    positional_encoding_configs = {
        #'2d_normalized': {'label_dim': 2},
        #'x_normalized': {'label_dim': 1},
        #'y_normalized': {'label_dim': 1},
        'dist_center': {'label_dim': 1}
    }

    for encoding_type, encoding_params in positional_encoding_configs.items():
        print(f"\nRunning input pixel experiments for encoding: {encoding_type}")
        current_config = base_config_copy.copy()
        label_dim = encoding_params['label_dim']

        for seed in range(num_seeds):
            print(f"  Seed: {seed}")
            setup_and_train_input_pixel(seed, encoding_type, label_dim, project_name=project_name, config=current_config)

def run_inputpixels_subsets(*, num_seeds=2, thickness=2, project_name="decoder-inputpixels-subsets"):
    """Radial vs. scrambled_radial (thickness = w).
    Args:
        num_seeds (int, optional): Number of random seeds to use. Defaults to 2.
        thickness (int, optional): Thickness parameter. Defaults to 2.
        project_name (str, optional): W&B project name.
    """
    current_base_config = base_config.copy()
    for seed in range(num_seeds):
        for subgraph_type in ("scrambled_radial", "radial"):
            setup_and_train_input_pixel(
                seed=seed,
                positional_encoding_type="dist_center",
                label_dim=1,
                project_name=project_name,
                config={
                    **current_base_config,
                    "subgraph_type": subgraph_type,
                    "subgraph_param": thickness,
                    "untrained": False,        # trained, no‑dropout nets
                },
            )

def run_similarity_comparison_classid(num_seeds=5, project_name="classid-tgt-similarity-comparison"):
    """
    Runs comparisons between FC and FC+Dropout models with and without using only
    the cosine similarity vector of the target as the input to the decoder.

    Args:
        num_seeds (int, optional): Number of random seeds to use. Defaults to 5.
        project_name (str, optional): W&B project name.
    """
    model_types = ['fully_connected', 'fully_connected_dropout']
    similarity_options = [True, False]

    for model_str in model_types:
        for use_similarity in similarity_options:
            print(f"\nRunning config: model={model_str}, use_target_similarity_only={use_similarity}")
            
            # Create specific config for this run
            current_config = base_config.copy()
            current_config["model_class_str"] = model_str
            current_config["varying_dim"] = False 
            current_config["untrained"] = False
            current_config["use_target_similarity_only"] = use_similarity
            
            for seed in range(num_seeds):
                print(f"  Seed: {seed}")
                # Call the class ID setup function
                setup_and_train_class_id(
                    seed=seed, 
                    num_neurons=current_config['num_neurons'], # Use num_neurons from config
                    project_name=project_name, 
                    config=current_config
                )

def run_mixed_hidden_dims_classid(num_seeds=5, project_name="classid-mixed-hidden-dims", 
                                 train_hidden_dim=None, valid_hidden_dim=None,
                                 train_samples=8000, valid_samples=2000):
    """
    Run experiments with different hidden dimensions for training and validation.

    Args:
        num_seeds (int, optional): Number of random seeds to use. Defaults to 5.
        project_name (str, optional): W&B project name.
        train_hidden_dim (list, optional): Hidden dimensions for training data. Defaults to [100].
        valid_hidden_dim (list, optional): Hidden dimensions for validation data. Defaults to [50, 50].
        train_samples (int, optional): Number of training samples. Defaults to 8000.
        valid_samples (int, optional): Number of validation samples. Defaults to 2000.
    """
    # Set default architectures if not provided
    if train_hidden_dim is None:
        train_hidden_dim = [100]
    if valid_hidden_dim is None:
        valid_hidden_dim = [50, 50]
        
    for seed in range(num_seeds):
        print(f"Running mixed hidden dims experiment - Seed: {seed}")
        print(f"  Train architecture: {train_hidden_dim}")
        print(f"  Valid architecture: {valid_hidden_dim}")
        
        # Create configs for training and validation datasets
        train_config = base_config.copy()
        train_config["hidden_dim"] = train_hidden_dim
        train_config["varying_dim"] = False
        train_config["untrained"] = False
        
        valid_config = base_config.copy()
        valid_config["hidden_dim"] = valid_hidden_dim
        valid_config["varying_dim"] = False
        valid_config["untrained"] = False
        
        setup_and_train_mixed_hidden_dims(
            seed=seed,
            train_config=train_config,
            valid_config=valid_config,
            train_samples=train_samples,
            valid_samples=valid_samples,
            project_name=project_name
        ) 