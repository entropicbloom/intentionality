import numpy as np
import torch
import os
import time
from datasets import OneLayerDataset, OneLayerDataModule
from set_transformer import SetTransformer
from torch.utils.data import DataLoader
from lightning_model import LightningModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

model_type = 'FullyConnected'
dataset_type = 'MNISTDataModule'

loader = DataLoader(OneLayerDataset(model_type, dataset_type, 2), batch_size=5, shuffle=True)
pytorch_model = SetTransformer(dim_input=50, num_outputs=1, dim_output=10, num_inds=16, dim_hidden=128, num_heads=4, ln=False)

lightning_model = LightningModel(pytorch_model, learning_rate=0.001, num_classes=10)
data_module = OneLayerDataModule(model_type, dataset_type, layer_idx=2, input_dim=50, batch_size=128, num_workers=0)

callbacks = [
        ModelCheckpoint(
            save_top_k=1, mode="max", monitor="valid_acc"
        )  # save top 1 model
    ]
logger = CSVLogger(save_dir="logs/", name="my-model")
trainer = pl.Trainer(
    max_epochs=100,
    callbacks=callbacks,
    accelerator="auto",  # Uses GPUs or TPUs if available
    devices="auto",  # Uses all available GPUs/TPUs if applicable
    logger=logger,
    deterministic=False,
    log_every_n_steps=10,
)

start_time = time.time()
trainer.fit(model=lightning_model, datamodule=data_module)

runtime = (time.time() - start_time) / 60
print(f"Training took {runtime:.2f} min in total.")
