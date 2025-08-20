import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# Assuming these are in the parent directories or installed packages
from underlying_datasets import DatasetClassificationDataModule
from lightning_model import LightningClassificationModel
from underlying.utils import get_dir_path
from decoder.models import decoder_dict

def setup_and_train_dataset_classification(seed, train_samples, valid_samples, project_name, config):
    """Sets up and trains a decoder model for dataset classification (MNIST vs Fashion-MNIST)."""
    torch.manual_seed(seed)

    # Initialize wandb with the provided project name
    wandb.init(
        project=project_name,
        config=config,
        name=f"dataset-classification-{config['decoder_class']}-s{seed}",
        group=f"dataset-classification-{config['decoder_class']}"
    )

    # Initialize model using decoder_dict from models.py
    # For dataset classification, we use cosine similarities of output neurons
    # 10x10 similarity matrix preserves neuron-by-neuron structure for self-attention
    pytorch_model = decoder_dict[config['decoder_class']](
        dim_input=10,   # 10 similarity features per neuron (10x10 matrix structure)
        num_outputs=1,
        dim_output=2,   # Binary classification: MNIST vs Fashion-MNIST
        num_inds=10,    # 10 neurons (output layer size)
        dim_hidden=64,
        num_heads=4,
        ln=False
    )

    # Setup training
    lightning_model = LightningClassificationModel(pytorch_model, learning_rate=0.001, num_classes=2)
    data_module = DatasetClassificationDataModule(
        batch_size=64,
        num_workers=0,
        train_samples=train_samples,
        valid_samples=valid_samples,
        config=config
    )

    # Training configuration
    callbacks = [ModelCheckpoint(save_top_k=1, mode="max", monitor="valid_acc")]
    trainer = pl.Trainer(
        max_epochs=100,
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