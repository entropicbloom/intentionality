import numpy as np
import torch
import os
from datasets import OneLayerDataset, OneLayerDataModule
from decoder import TransformerDecoder, FCDecoder
from lightning_model import LightningModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

# Configuration dictionary
config = {
    "model_type": 'FullyConnectedDropout',
    "dataset_type": 'MNISTDataModule',
    "decoder_class": 'TransformerDecoder',
    "preprocessing": 'multiply_transpose',
    "untrained": False,
    "varying_dim": False,
}

# Model mapping
decoder_dict = {
    'FCDecoder': FCDecoder,
    'Transformer': TransformerDecoder,
}

def run(seed):
    torch.manual_seed(seed)
    
    # Build configuration string
    untrained_str = '-untrained' if config['untrained'] else ''
    varying_dim_str = '-varying-dim' if config['varying_dim'] else ''
    underlying_config_str = f"{config['model_type']}-{config['dataset_type']}{untrained_str}{varying_dim_str}"
    dataset_path = f'../underlying/saved_models/{underlying_config_str}/'

    # Initialize wandb
    wandb.init(
        project="decoder-4",
        config=config,
        name=f"{underlying_config_str}-{config['decoder_class']}-{seed}"
    )

    # Initialize model
    pytorch_model = decoder_dict[config['decoder_class']](
        dim_input=10,
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
        preprocessing=config['preprocessing']
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
    for seed in range(2, 5):
        run(seed)