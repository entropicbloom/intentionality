import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import os

from datasets.CIFAR import CIFARDataModule
from datasets.MNIST import MNISTDataModule
from pytorch_models.alexnet import AlexNet
from pytorch_models.fully_connected import FullyConnected
from lightning_model import LightningModel

def run(model_class, dataset_class, batch_size, num_epochs, learning_rate, num_workers, num_classes, seed):
    torch.manual_seed(seed) 
    data_module = dataset_class(batch_size, num_workers, data_path='./data')
    
    #pytorch_model = AlexNet(num_classes=num_classes)
    pytorch_model = model_class(num_classes=num_classes, input_dim=data_module.input_dim, hidden_dim=[50,50])

    lightning_model = LightningModel(pytorch_model, learning_rate=learning_rate, num_classes=num_classes)

    callbacks = [
        ModelCheckpoint(
            save_top_k=1, mode="max", monitor="valid_acc"
        )  # save top 1 model
    ]
    logger = CSVLogger(save_dir="logs/", name="my-model")

    # %load ../code_lightningmodule/trainer_nb_basic.py
    import time


    trainer = pl.Trainer(
        max_epochs=num_epochs,
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

    path = f'saved_models/{model_class.__name__}-{dataset_class.__name__}/'
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(pytorch_model.state_dict(), path + f'seed-{seed}')


if __name__ == '__main__':

    for seed in range(551, 1000):
        run(
            FullyConnected,
            MNISTDataModule,
            batch_size=256,
            num_epochs=2,
            learning_rate=0.001,
            num_workers=4,
            num_classes=10,
            seed=seed
        )
