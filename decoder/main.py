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
    "model_class_str": 'fully_connected_dropout',  # was previously "model_type"
    "dataset_class_str": 'mnist',                     # was previously "dataset_type"
    "decoder_class": 'TransformerDecoder',
    "preprocessing": 'multiply_transpose',
    "untrained": False,
    "varying_dim": False,
    "num_neurons": 10,  # New parameter to specify max number of neurons
    "min_neurons": 2,   # New parameter to specify min number of neurons
}

# Model mapping
decoder_dict = {
    'FCDecoder': FCDecoder,
    'TransformerDecoder': TransformerDecoder,
}

MODELS_DIR ='saved_models/'

# Import the path creation function from underlying
from underlying.utils import get_dir_path

def run(seed, num_neurons):
    torch.manual_seed(seed)
    
    # Use get_dir_path to create the dataset path
    # Note: num_epochs=0 if config['untrained'] else 1 to match the untrained logic
    dataset_path = '../underlying/' + get_dir_path(
        model_class_str=config['model_class_str'],
        dataset_class_str=config['dataset_class_str'],
        num_epochs=0 if config['untrained'] else 1,
        varying_dim=config['varying_dim'], 
        models_dir=MODELS_DIR
    )

    # Get the configuration string for wandb naming
    underlying_config_str = dataset_path.split('/')[-2]  # Extract the directory name
    
    # Initialize wandb with a new project name
    wandb.init(
        project="decoder-neuron-ablation",  # Changed from "decoder" to "decoder-neuron-ablation"
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

if __name__ == '__main__':
    # Loop through different numbers of neurons
    for num_neurons in range(config['min_neurons'], config['num_neurons'] + 1):
        print(f"Running experiments with {num_neurons} neurons")
        for seed in range(1):
            run(seed, num_neurons)