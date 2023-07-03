import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from datasets.CIFAR import CIFARDataModule
from pytorch_models.alexnet import AlexNet
from lightning_model import LightningModel

if __name__ == '__main__':

    BATCH_SIZE = 256
    NUM_EPOCHS = 40
    LEARNING_RATE = 0.0001
    NUM_WORKERS = 1
    NUM_CLASSES = 10
    SEED = 1

    torch.manual_seed(SEED) 
    data_module = CIFARDataModule(BATCH_SIZE, NUM_WORKERS, data_path='./data')
    pytorch_model = AlexNet(num_classes=NUM_CLASSES)

    lightning_model = LightningModel(pytorch_model, learning_rate=LEARNING_RATE, num_classes=NUM_CLASSES)

    callbacks = [
        ModelCheckpoint(
            save_top_k=1, mode="max", monitor="valid_acc"
        )  # save top 1 model
    ]
    logger = CSVLogger(save_dir="logs/", name="my-model")

    # %load ../code_lightningmodule/trainer_nb_basic.py
    import time


    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
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