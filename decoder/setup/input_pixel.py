# decoder/experiments/input_pixel.py
import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import os
import uuid
import json

# Assuming these are in the parent directories or installed packages
from underlying_datasets import FirstLayerDataModule, MixedDatasetFirstLayerDataModule
from lightning_model import LightningRegressionModel
from underlying.utils import get_dir_path
from decoder.models import decoder_dict # Changed back to absolute import

def save_decoder_model(pytorch_model, config, positional_encoding_type, seed, label_dim, 
                      actual_dim_input, **extra_metadata):
    """Helper function to save decoder models with UUID directory names and config."""
    model_uuid = str(uuid.uuid4())
    decoder_save_dir = f"decoder_models/{model_uuid}"
    os.makedirs(decoder_save_dir, exist_ok=True)
    
    # Save model
    model_save_path = f"{decoder_save_dir}/model.pt"
    torch.save(pytorch_model.state_dict(), model_save_path)
    
    # Save config with additional metadata
    save_config = config.copy()
    save_config.update({
        "uuid": model_uuid,
        "positional_encoding_type": positional_encoding_type,
        "seed": seed,
        "label_dim": label_dim,
        "actual_dim_input": actual_dim_input,
        **extra_metadata
    })
    config_save_path = f"{decoder_save_dir}/config.json"
    with open(config_save_path, 'w') as f:
        json.dump(save_config, f, indent=2)
    
    print(f"Saved decoder model to: {decoder_save_dir}")
    print(f"Model UUID: {model_uuid}")
    return model_uuid

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

    # Setup data module first to determine actual input dimensions
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
    
    # Setup the dataset to get actual input dimensions
    data_module.setup()
    
    # Get actual input size by checking the similarity matrix from the dataset
    sample_sim, _ = data_module.train_set.dataset[0]
    actual_dim_input = sample_sim.shape[1]  # width of similarity matrix

    # Initialize model using decoder_dict from models.py
    pytorch_model = decoder_dict[config['decoder_class']]( 
        dim_input=actual_dim_input, # Use actual similarity matrix size
        num_outputs=1, # Predicting property of one pixel at a time
        dim_output=label_dim, # Dimension of the positional encoding
        num_inds=16,
        dim_hidden=64,
        num_heads=4,
        ln=False
    )

    # Loss needs to handle regression (MSE) instead of classification
    lightning_model = LightningRegressionModel(pytorch_model, learning_rate=0.001, label_dim=label_dim)

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
    
    # Save the trained decoder model
    save_decoder_model(pytorch_model, config, positional_encoding_type, seed, 
                      label_dim, actual_dim_input)
    
    wandb.finish()

def setup_and_train_mixed_datasets(seed, train_config, valid_config, positional_encoding_type, 
                                  label_dim, train_samples, valid_samples, project_name):
    """Sets up and trains a decoder model for input pixel decoding with different underlying datasets for train and validation."""
    torch.manual_seed(seed)

    # Get dataset paths for both configurations
    train_dataset_path = '../underlying/' + get_dir_path(
        model_class_str=train_config['model_class_str'],
        dataset_class_str=train_config['dataset_class_str'],
        num_epochs=0 if train_config['untrained'] else 2,
        hidden_dim=train_config['hidden_dim'],
        varying_dim=train_config['varying_dim'],
        models_dir=train_config['models_dir']
    )
    
    valid_dataset_path = '../underlying/' + get_dir_path(
        model_class_str=valid_config['model_class_str'],
        dataset_class_str=valid_config['dataset_class_str'],
        num_epochs=0 if valid_config['untrained'] else 2,
        hidden_dim=valid_config['hidden_dim'],
        varying_dim=valid_config['varying_dim'],
        models_dir=valid_config['models_dir']
    )

    # Get configuration strings for wandb naming
    train_config_str = train_dataset_path.split('/')[-2]
    valid_config_str = valid_dataset_path.split('/')[-2]

    # Update config for logging this specific run
    run_config = {
        "train_config": train_config,
        "valid_config": valid_config,
        "positional_encoding_type": positional_encoding_type,
        "seed": seed,
        "label_dim": label_dim,
        "experiment_type": "input_pixels_mixed_datasets",
        "train_samples": train_samples,
        "valid_samples": valid_samples,
        "decoder_class": train_config['decoder_class']
    }

    # Initialize wandb with the provided project name
    # Shorten wandb names to avoid 128 char limit
    decoder_short = train_config['decoder_class'].replace('TransformerDecoder', 'TD').replace('_dropout', '_dr')
    train_model = train_config['model_class_str'].replace('fully_connected', 'fc')
    valid_model = valid_config['model_class_str'].replace('fully_connected', 'fc')
    wandb_name = f"mixed-{train_model}-{valid_model}-{decoder_short}-{positional_encoding_type}"
    wandb_group = f"mixed-{train_model}-{valid_model}-{decoder_short}-{positional_encoding_type}"
    # Optionally add suffix for target similarity only
    if train_config.get('use_target_similarity_only', False):
        wandb_name += "-target_sim"
        wandb_group += "-target_sim"
    wandb_name += f"-s{seed}"

    wandb.init(
        project=project_name,
        config=run_config,
        name=wandb_name,
        group=wandb_group
    )

    # Setup data module with mixed datasets
    data_module = MixedDatasetFirstLayerDataModule(
        train_dataset_path=train_dataset_path,
        valid_dataset_path=valid_dataset_path,
        positional_encoding_type=positional_encoding_type,
        batch_size=64,
        num_workers=0,
        # Extract subgraph parameters from config if they exist
        subgraph_type=train_config.get("subgraph_type"),
        subgraph_param=train_config.get("subgraph_param"),
        use_target_similarity_only=train_config.get('use_target_similarity_only', False),
        train_samples=train_samples,
        valid_samples=valid_samples
    )
    
    # Setup the dataset to get actual input dimensions
    data_module.setup()
    
    # Get actual input size by checking the similarity matrix from the dataset
    sample_sim, _ = data_module.train_set.dataset[0]
    actual_dim_input = sample_sim.shape[1]  # width of similarity matrix

    # Initialize model using decoder_dict from models.py
    pytorch_model = decoder_dict[train_config['decoder_class']]( 
        dim_input=actual_dim_input, # Use actual similarity matrix size
        num_outputs=1, # Predicting property of one pixel at a time
        dim_output=label_dim, # Dimension of the positional encoding
        num_inds=16,
        dim_hidden=64,
        num_heads=4,
        ln=False
    )

    # Loss needs to handle regression (MSE) instead of classification
    lightning_model = LightningRegressionModel(pytorch_model, learning_rate=0.001, label_dim=label_dim)

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
    
    # Save the trained decoder model (only train_config affects decoder weights)
    save_decoder_model(pytorch_model, train_config, positional_encoding_type, seed, 
                      label_dim, actual_dim_input, train_samples=train_samples)
    
    wandb.finish()

def setup_and_eval(model_uuid: str, eval_dataset_path: str, batch_size: int = 64, num_workers: int = 0):
    """
    Load a saved decoder model by UUID and evaluate it on a specified dataset.
    
    Args:
        model_uuid: UUID of the saved decoder model
        eval_dataset_path: Path to the evaluation dataset
        batch_size: Batch size for evaluation
        num_workers: Number of workers for data loading
        
    Returns:
        Dict containing evaluation metrics
    """
    # Load saved model configuration
    config_path = f"decoder_models/{model_uuid}/config.json"
    with open(config_path, 'r') as f:
        saved_config = json.load(f)
    
    # Extract configuration parameters
    decoder_class = saved_config['decoder_class']
    positional_encoding_type = saved_config['positional_encoding_type']
    label_dim = saved_config['label_dim']
    actual_dim_input = saved_config['actual_dim_input']
    
    # Reconstruct the decoder model with same parameters as training
    pytorch_model = decoder_dict[decoder_class](
        dim_input=actual_dim_input,
        num_outputs=1,
        dim_output=label_dim,
        num_inds=16,
        dim_hidden=64,
        num_heads=4,
        ln=False
    )
    
    # Load the trained weights
    model_path = f"decoder_models/{model_uuid}/model.pt"
    pytorch_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    
    # Wrap in Lightning model
    lightning_model = LightningRegressionModel(pytorch_model, learning_rate=0.001, label_dim=label_dim)
    
    # Setup evaluation data module using same configuration as training
    data_module = FirstLayerDataModule(
        eval_dataset_path,
        positional_encoding_type=positional_encoding_type,
        batch_size=batch_size,
        num_workers=num_workers,
        subgraph_type=saved_config.get("subgraph_type"),
        subgraph_param=saved_config.get("subgraph_param"),
        use_target_similarity_only=saved_config.get('use_target_similarity_only', False),
    )
    
    # Setup the dataset
    data_module.setup()
    
    # Create trainer for evaluation only
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        logger=False,  # No logging for evaluation
        enable_progress_bar=True,
        enable_model_summary=False,
        limit_val_batches=0.1  # Match training config - only use 10% of validation data
    )
    
    # Run evaluation using validation dataloader
    print(f"Evaluating model {model_uuid} on dataset: {eval_dataset_path}")
    validation_results = trainer.validate(model=lightning_model, datamodule=data_module)
    
    return {
        "model_uuid": model_uuid,
        "eval_dataset_path": eval_dataset_path,
        "results": validation_results[0] if validation_results else {},
        "config": saved_config
    } 