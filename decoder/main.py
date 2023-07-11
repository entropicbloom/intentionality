import numpy as np
import torch
import os
import time
from datasets import OneLayerDataset, OneLayerDataModule
from decoder import Transformer, FCDecoder
from torch.utils.data import DataLoader
from lightning_model import LightningModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
import wandb

config = {
    "model_type": 'FullyConnectedDropout',
    "dataset_type": 'MNISTDataModule',
    "decoder_class": 'FCDecoder',
}

decoder_dict = {
    'FCDecoder': FCDecoder,
    'Transformer': Transformer,
}

def run(seed):
    torch.manual_seed(seed) 
    wandb.init(project="decoder", config=config, name=f"{config['model_type']}-{config['dataset_type']}-{config['decoder_class']}-{seed}")

    pytorch_model = decoder_dict[config['decoder_class']](dim_input=10, num_outputs=1, dim_output=10, num_inds=16, dim_hidden=64, num_heads=4, ln=False)

    lightning_model = LightningModel(pytorch_model, learning_rate=0.001, num_classes=10)
    data_module = OneLayerDataModule(config['model_type'], config['dataset_type'], layer_idx=2, input_dim=50, batch_size=128, num_workers=0)

    callbacks = [
            ModelCheckpoint(
                save_top_k=1, mode="max", monitor="valid_acc"
            )  # save top 1 model
        ]

    wandb_logger = WandbLogger()
    trainer = pl.Trainer(
        max_epochs=50,
        callbacks=callbacks,
        accelerator="auto",  # Uses GPUs or TPUs if available
        devices="auto",  # Uses all available GPUs/TPUs if applicable
        deterministic=False,
        log_every_n_steps=10,
        logger=wandb_logger
    )

    trainer.fit(model=lightning_model, datamodule=data_module)

    wandb.finish()

if __name__ == '__main__':

    for seed in range(5):
        run(seed)