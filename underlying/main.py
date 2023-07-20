import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import os

from datasets.CIFAR import CIFARDataModule
from datasets.MNIST import MNISTDataModule
from pytorch_models.alexnet import AlexNet
from pytorch_models.fully_connected import FullyConnected, FullyConnectedDropout, FullyConnectedGenerative, FullyConnectedGenerativeDropout
from lightning_model import LightningModel

def get_dir_path(model_class, dataset_class, num_epochs, varying_dim):
    untrained_str = '-untrained' if num_epochs == 0 else ''
    varying_dim_str = '-varying-dim' if varying_dim else ''
    path = f'saved_models/{model_class.__name__}-{dataset_class.__name__}{untrained_str}{varying_dim_str}/'
    return path

def run(model_class, dataset_class, batch_size, num_epochs, learning_rate, num_workers, num_classes,
        hidden_dim, seed, varying_dim_bounds=None):
    torch.manual_seed(seed) 
    
    # initialize data module
    data_module = dataset_class(batch_size, num_workers, data_path='./data')
    
    # initialize model
    pytorch_model = model_class(num_classes=num_classes, input_dim=data_module.input_dim, hidden_dim=hidden_dim)
    lightning_model = LightningModel(pytorch_model, learning_rate=learning_rate, num_classes=num_classes)

    callbacks = [
        ModelCheckpoint(
            save_top_k=1, mode="max", monitor="valid_acc"
        )  # save top 1 model
    ]
    logger = CSVLogger(save_dir="logs/", name="my-model")


    trainer = pl.Trainer(
        max_epochs=num_epochs,
        callbacks=callbacks,
        accelerator="auto",  # Uses GPUs or TPUs if available
        devices="auto",  # Uses all available GPUs/TPUs if applicable
        logger=logger,
        deterministic=False,
        log_every_n_steps=10,
    )

    if num_epochs > 0:
        trainer.fit(model=lightning_model, datamodule=data_module)

    path = get_dir_path(model_class, dataset_class, num_epochs, varying_dim_bounds)
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(pytorch_model.state_dict(), path + f'seed-{seed}')


if __name__ == '__main__':

    train_config = {
        'model_class': FullyConnected,
        'dataset_class': MNISTDataModule,
        'batch_size': 256,
        'num_epochs': 0,
        'learning_rate': 0.001,
        'num_workers': 4,
        'num_classes': 10,
        'hidden_dim': [50,50],
        'varying_dim_bounds': None#(25, 100)
    }

    for seed in range(1000):

        # vary hidden dimension if necessary
        if train_config['varying_dim_bounds'] is not None:
            random_dimension = np.random.randint(*train_config['varying_dim_bounds'])
            train_config['hidden_dim'] = [random_dimension] * len(train_config['hidden_dim'])

        # train underlying model
        run(
            **train_config,
            seed=seed
        )

    # save config
    path = get_dir_path(train_config['model_class'], train_config['dataset_class'], train_config['num_epochs'], None)
    with open(path + 'train_config.txt', 'w') as f:
        print(train_config, file=f)
