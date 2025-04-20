# decoder/experiments/input_pixel.py
import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# Assuming these are in the parent directories or installed packages
from underlying_datasets import FirstLayerDataModule
from lightning_model import LightningRegressionModel
from underlying.utils import get_dir_path
from decoder.models import decoder_dict # Changed back to absolute import

def setup_and_train(seed, positional_encoding_type, label_dim, project_name, config):
    """Sets up and trains a decoder model for input pixel decoding."""
    torch.manual_seed(seed)

    # Use get_dir_path to create the dataset path (reverted)
    # Layer 0 weights are needed, these models should have been trained normally.
    dataset_path = '../underlying/' + get_dir_path(
        model_class_str=config['model_class_str'],
        dataset_class_str=config['dataset_class_str'],
        num_epochs=0 if config['untrained'] else 2,
        hidden_dim=config['hidden_dim'],
        varying_dim=config['varying_dim'],
        models_dir=config['models_dir']
    )

    # Get the configuration string for wandb naming (reverted)
    underlying_config_str = dataset_path.split('/')[-2]  # Extract the directory name

    # Update config for logging this specific run
    run_config = config.copy()
    run_config['positional_encoding_type'] = positional_encoding_type
    run_config['seed'] = seed
    run_config['label_dim'] = label_dim
    run_config['experiment_type'] = 'input_pixels'

    # Initialize wandb with the provided project name
    wandb_name = f"{underlying_config_str}-{config['decoder_class']}-{positional_encoding_type}"
    wandb_group = f"{underlying_config_str}-{config['decoder_class']}-{positional_encoding_type}"
    # Optionally add suffix for target similarity only
    if run_config.get('use_target_similarity_only', False): # Use .get for safety
        wandb_name += "-target_sim"
        wandb_group += "-target_sim"
    wandb_name += f"-s{seed}"

    wandb.init(
        project=project_name,
        config=run_config,
        name=wandb_name,
        group=wandb_group
    )

    # Initialize model using decoder_dict from models.py
    pytorch_model = decoder_dict[config['decoder_class']]( 
        dim_input=784, # Number of input pixels
        num_outputs=1, # Predicting property of one pixel at a time
        dim_output=label_dim, # Dimension of the positional encoding
        num_inds=16,
        dim_hidden=64,
        num_heads=4,
        ln=False
    )

    # Setup training using FirstLayerDataModule
    # Loss needs to handle regression (MSE) instead of classification
    lightning_model = LightningRegressionModel(pytorch_model, learning_rate=0.001, label_dim=label_dim)
    data_module = FirstLayerDataModule(
        dataset_path,
        positional_encoding_type=positional_encoding_type,
        batch_size=64,
        num_workers=0,
        # Extract subgraph parameters from config if they exist
        subgraph_type=config.get("subgraph_type"),
        subgraph_param=config.get("subgraph_param"),
        use_target_similarity_only=config.get('use_target_similarity_only', False),
    )

    # Training configuration
    # Monitor validation loss (MSE) instead of accuracy
    # Checkpoint based on the validation metric (mse)
    callbacks = [ModelCheckpoint(save_top_k=1, mode="min", monitor="valid_mse")]
    trainer = pl.Trainer(
        max_epochs=4,
        val_check_interval=500,
        limit_val_batches=0.1,
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