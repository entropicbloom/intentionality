import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# Assuming these are in the parent directories or installed packages
from underlying_datasets import LastLayerDataModule
from lightning_model import LightningClassificationModel
from underlying.utils import get_dir_path
from decoder.models import decoder_dict # Changed back to absolute import

def setup_and_train(seed, num_neurons, project_name, config):
    """Sets up and trains a decoder model for class ID prediction."""
    torch.manual_seed(seed)

    # Use get_dir_path to create the dataset path (reverted)
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

    # Initialize wandb with the provided project name
    wandb.init(
        project=project_name,
        config=config,
        name=f"{underlying_config_str}-{config['decoder_class']}-n{num_neurons}-s{seed}",
        group=f"{underlying_config_str}-{config['decoder_class']}-n{num_neurons}"
    )

    # Create a list of neuron indices to use
    use_neurons = list(range(num_neurons))

    # Initialize model using decoder_dict from models.py
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
    lightning_model = LightningClassificationModel(pytorch_model, learning_rate=0.001, num_classes=10)
    data_module = LastLayerDataModule(
        dataset_path, 
        layer_idx=2,
        input_dim=50,
        batch_size=64,
        num_workers=0,
        transpose_weights=False,
        preprocessing=config['preprocessing'],
        use_neurons=use_neurons,  # Pass the list of neurons to use
        use_target_similarity_only=config.get('use_target_similarity_only', False),
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