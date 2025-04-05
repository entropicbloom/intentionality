import numpy as np
import torch
import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from datasets import OneLayerDataset, OneLayerDataModule
from decoder import TransformerDecoder, FCDecoder
from lightning_model import LightningModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

# Updated Configuration dictionary to align with underlying/main.py naming conventions
config = {
    "model_class_str": 'fully_connected_dropout',
    "dataset_class_str": 'mnist',
    "decoder_class": 'TransformerDecoder',
    "preprocessing": 'multiply_transpose',
    "untrained": False,
    "varying_dim": False,
    "num_neurons": 10,
    "min_neurons": 2,
}

# Model mapping
decoder_dict = {
    'FCDecoder': FCDecoder,
    'TransformerDecoder': TransformerDecoder,
}

MODELS_DIR ='saved_models/'

# Import the path creation function from underlying
from underlying.utils import get_dir_path

def run(seed, num_neurons, project_name, config):
    torch.manual_seed(seed)
    
    # Use get_dir_path to create the dataset path
    # Note: num_epochs=0 if config['untrained'] else 1 to match the untrained logic
    dataset_path = '../underlying/' + get_dir_path(
        model_class_str=config['model_class_str'],
        dataset_class_str=config['dataset_class_str'],
        num_epochs=0 if config['untrained'] else 2,
        hidden_dim=config['hidden_dim'],
        varying_dim=config['varying_dim'], 
        models_dir=MODELS_DIR
    )

    # Get the configuration string for wandb naming
    underlying_config_str = dataset_path.split('/')[-2]  # Extract the directory name
    
    # Initialize wandb with the provided project name
    wandb.init(
        project=project_name,
        config=config,
        name=f"{underlying_config_str}-{config['decoder_class']}-n{num_neurons}-s{seed}",
        group=f"{underlying_config_str}-{config['decoder_class']}-n{num_neurons}"
    )

    # Create a list of neuron indices to use
    use_neurons = list(range(num_neurons))
    
    # Initialize model
    pytorch_model = decoder_dict[config['decoder_class']](
        dim_input=num_neurons,  # Update the input dimension to match number of neurons
        num_outputs=1,
        dim_output=10,
        num_inds=16,
        dim_hidden=64,
        num_heads=4,
        ln=False
    )

    # Setup training
    lightning_model = LightningModel(pytorch_model, learning_rate=0.001, num_classes=10)
    data_module = OneLayerDataModule(
        dataset_path,
        layer_idx=2,
        input_dim=50,
        batch_size=64,
        num_workers=0,
        transpose_weights=False,
        preprocessing=config['preprocessing'],
        use_neurons=use_neurons  # Pass the list of neurons to use
    )

    # Training configuration
    callbacks = [ModelCheckpoint(save_top_k=1, mode="max", monitor="valid_acc")]
    trainer = pl.Trainer(
        max_epochs=200,
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
        deterministic=False,
        log_every_n_steps=10,
        logger=WandbLogger()
    )

    # Train model
    trainer.fit(model=lightning_model, datamodule=data_module)
    wandb.finish()

def run_ablation_experiments(min_neurons=None, max_neurons=None, num_seeds=5, experiment_config=None):
    """
    Run ablation experiments by varying the number of neurons.
    
    Args:
        min_neurons (int, optional): Minimum number of neurons to use. Defaults to config value.
        max_neurons (int, optional): Maximum number of neurons to use. Defaults to config value.
        num_seeds (int, optional): Number of random seeds to use. Defaults to 5.
        experiment_config (dict, optional): Configuration to use. Defaults to global config.
    """
    # Use global config if no config is provided
    if experiment_config is None:
        experiment_config = config.copy()
    
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
            run(seed, num_neurons, project_name="decoder-neuron-ablation", config=current_config)

def run_main_experiments(num_seeds=5):
    """
    Run experiments with all neurons for different configurations:
    1. The current config
    2. A config with model_class_str='fully_connected'
    3. A config with untrained=True
    4. A config with varying_dim=True
    
    Each configuration is run with all neurons for multiple seeds.
    
    Args:
        num_seeds (int, optional): Number of random seeds to use. Defaults to 5.
    """
    # Save the original config
    original_config = config.copy()
    
    # First run with the current config
    print("Running with current config")
    current_config = original_config.copy()
    for seed in range(num_seeds):
        run(seed, current_config['num_neurons'], project_name="decoder-main-experiments", config=current_config)
    
    # Run with model_class_str='fully_connected'
    fc_config = original_config.copy()
    fc_config["model_class_str"] = 'fully_connected'
    print(f"Running with model_class_str={fc_config['model_class_str']}")
    for seed in range(num_seeds):
        run(seed, fc_config['num_neurons'], project_name="decoder-main-experiments", config=fc_config)
    
    # Run with untrained=True, model_class_str='fully_connected'
    untrained_config = original_config.copy()
    untrained_config["untrained"] = True
    untrained_config["model_class_str"] = 'fully_connected'
    print(f"Running with untrained={untrained_config['untrained']}")
    for seed in range(num_seeds):
        run(seed, untrained_config['num_neurons'], project_name="decoder-main-experiments", config=untrained_config)
    
    # Run with varying_dim=True
    varying_dim_config = original_config.copy()
    varying_dim_config["varying_dim"] = True
    print(f"Running with varying_dim={varying_dim_config['varying_dim']}")
    for seed in range(num_seeds):
        run(seed, varying_dim_config['num_neurons'], project_name="decoder-main-experiments", config=varying_dim_config)

if __name__ == '__main__':
    run_ablation_experiments()