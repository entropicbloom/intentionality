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

# Training Constants
MAX_SEEDS = 1000
LOG_STEPS = 10

# Default Training Configuration
CONFIG = {
    'model_class': FullyConnectedDropout,
    'dataset_class': MNISTDataModule,
    'batch_size': 256,
    'num_epochs': 2,
    'learning_rate': 0.001,
    'num_workers': 4,
    'num_classes': 10,
    'hidden_dim': [50, 50],
    'varying_dim_bounds': (25, 100)
}

# File System Constants
DATA_DIR = './data'
LOGS_DIR = 'logs/'
MODELS_DIR = 'saved_models/'

def get_dir_path(model_class, dataset_class, num_epochs, varying_dim):
    untrained_str = '-untrained' if num_epochs == 0 else ''
    varying_dim_str = '-varying-dim' if varying_dim else ''
    path = f'{MODELS_DIR}{model_class.__name__}-{dataset_class.__name__}{untrained_str}{varying_dim_str}/'
    return path

def run(model_class, dataset_class, batch_size, num_epochs, learning_rate, num_workers, num_classes,
        hidden_dim, seed, varying_dim_bounds=None):
    torch.manual_seed(seed) 
    
    # initialize data module
    data_module = dataset_class(batch_size, num_workers, data_path=DATA_DIR)
    
    # initialize model
    pytorch_model = model_class(num_classes=num_classes, input_dim=data_module.input_dim, hidden_dim=hidden_dim)
    lightning_model = LightningModel(pytorch_model, learning_rate=learning_rate, num_classes=num_classes)

    callbacks = [
        ModelCheckpoint(
            save_top_k=1, mode="max", monitor="valid_acc"
        )  # save top 1 model
    ]
    logger = CSVLogger(save_dir=LOGS_DIR, name="my-model")

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
        logger=logger,
        deterministic=False,
        log_every_n_steps=LOG_STEPS,
    )

    if num_epochs > 0:
        trainer.fit(model=lightning_model, datamodule=data_module)

    path = get_dir_path(model_class, dataset_class, num_epochs, varying_dim_bounds)
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(pytorch_model.state_dict(), path + f'seed-{seed}')

def get_start_seed(path):
    """
    Determine the next available seed number by checking existing files in the path.
    
    Args:
        path (str): Directory path to check for existing seed files
        
    Returns:
        int: Next available seed number
    """
    existing_seeds = []
    for filename in os.listdir(path):
        if filename.startswith('seed-'):
            try:
                seed_num = int(filename.split('seed-')[1])
                existing_seeds.append(seed_num)
            except ValueError:
                continue
    
    return max(existing_seeds) + 1 if existing_seeds else 0

def load_config(path):
    """Load existing config from file."""
    try:
        with open(path + 'train_config.txt', 'r') as f:
            # Using eval to convert string representation of dict back to dict
            return eval(f.read())
    except FileNotFoundError:
        return None

if __name__ == '__main__':
    train_config = CONFIG.copy()

    # Get the path and create directory if it doesn't exist
    path = get_dir_path(train_config['model_class'], train_config['dataset_class'], 
                       train_config['num_epochs'], train_config['varying_dim_bounds'])
    if not os.path.exists(path):
        os.makedirs(path)

    # Check if config exists and matches
    existing_config = load_config(path)
    if existing_config is not None:
        if existing_config != train_config:
            raise ValueError(
                f"Existing config in {path} does not match current config.\n"
                f"Existing: {existing_config}\n"
                f"Current: {train_config}"
            )
    else:
        # Save config if it doesn't exist
        with open(path + 'train_config.txt', 'w') as f:
            print(train_config, file=f)

    start_seed = get_start_seed(path)
    print(f"Starting training from seed {start_seed}")

    for seed in range(start_seed, MAX_SEEDS):
        # vary hidden dimension if necessary
        if train_config['varying_dim_bounds'] is not None:
            random_dimension = np.random.randint(*train_config['varying_dim_bounds'])
            train_config['hidden_dim'] = [random_dimension] * len(train_config['hidden_dim'])

        # train underlying model
        run(
            **train_config,
            seed=seed
        )
