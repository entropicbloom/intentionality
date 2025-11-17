"""
Training script for underlying models that saves checkpoints at specific epochs.
This is used for the "Training Dynamics of Intentionality" experiment.
"""
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import os
import sys
from utils import get_dir_path

from datasets.CIFAR import CIFARDataModule
from datasets.MNIST import MNISTDataModule
from datasets.FashionMNIST import FashionMNISTDataModule
from pytorch_models.alexnet import AlexNet
from pytorch_models.fully_connected import FullyConnected, FullyConnectedDropout, FullyConnectedGenerative, FullyConnectedGenerativeDropout
from lightning_model import LightningModel

# Class mappings
MODEL_MAP = {
    'fully_connected': FullyConnected,
    'fully_connected_dropout': FullyConnectedDropout,
    'fully_connected_generative': FullyConnectedGenerative,
    'fully_connected_generative_dropout': FullyConnectedGenerativeDropout,
    'alexnet': AlexNet
}

DATASET_MAP = {
    'mnist': MNISTDataModule,
    'cifar': CIFARDataModule,
    'fashionmnist': FashionMNISTDataModule
}

# Checkpoints to save at these epochs
CHECKPOINT_EPOCHS = [0, 1, 2, 3, 5, 10, 20, 50, 100]


class EpochCheckpointCallback(Callback):
    """
    Callback to save model checkpoints at specific epochs.
    """
    def __init__(self, checkpoint_epochs, save_dir, seed):
        super().__init__()
        self.checkpoint_epochs = checkpoint_epochs
        self.save_dir = save_dir
        self.seed = seed

        # Create checkpoint directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        """Save checkpoint at specific epochs."""
        current_epoch = trainer.current_epoch

        if current_epoch in self.checkpoint_epochs:
            checkpoint_path = os.path.join(
                self.save_dir,
                f'seed-{self.seed}_epoch-{current_epoch}'
            )
            # Save the underlying PyTorch model's state dict
            torch.save(pl_module.model.state_dict(), checkpoint_path)
            print(f"  Saved checkpoint at epoch {current_epoch}")


def train_with_checkpoints(
    model_class_str,
    dataset_class_str,
    batch_size,
    num_epochs,
    learning_rate,
    num_workers,
    num_classes,
    hidden_dim,
    seed,
    data_dir='./data',
    models_dir='saved_models_checkpoints/',
    checkpoint_epochs=None,
    varying_dim_bounds=None
):
    """
    Train a model and save checkpoints at specific epochs.

    Args:
        model_class_str: String identifier for model type
        dataset_class_str: String identifier for dataset
        batch_size: Batch size for training
        num_epochs: Total number of epochs to train
        learning_rate: Learning rate
        num_workers: Number of data loading workers
        num_classes: Number of output classes
        hidden_dim: Hidden layer dimensions
        seed: Random seed
        data_dir: Directory for dataset storage
        models_dir: Directory for saving checkpoints
        checkpoint_epochs: List of epochs to save checkpoints at
        varying_dim_bounds: Bounds for varying hidden dimensions
    """
    torch.manual_seed(seed)

    # Default checkpoint epochs if not provided
    if checkpoint_epochs is None:
        checkpoint_epochs = CHECKPOINT_EPOCHS

    # Only save checkpoints that are <= num_epochs
    checkpoint_epochs = [e for e in checkpoint_epochs if e <= num_epochs]

    # Get actual classes from string identifiers
    model_class = MODEL_MAP[model_class_str]
    dataset_class = DATASET_MAP[dataset_class_str]

    # Initialize data module
    data_module = dataset_class(batch_size, num_workers, data_path=data_dir)

    # Initialize model
    pytorch_model = model_class(
        num_classes=num_classes,
        input_dim=data_module.input_dim,
        hidden_dim=hidden_dim
    )
    lightning_model = LightningModel(
        pytorch_model,
        learning_rate=learning_rate,
        num_classes=num_classes
    )

    # Get save directory path
    save_path = get_dir_path(
        model_class_str,
        dataset_class_str,
        num_epochs,
        hidden_dim,
        varying_dim_bounds,
        models_dir
    )

    # Add callbacks
    callbacks = [
        EpochCheckpointCallback(
            checkpoint_epochs=checkpoint_epochs,
            save_dir=save_path,
            seed=seed
        )
    ]

    logger = CSVLogger(save_dir='logs/', name="checkpoint-training")

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
        logger=logger,
        deterministic=False,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    # Save initial checkpoint (epoch 0) before training
    if 0 in checkpoint_epochs:
        initial_checkpoint_path = os.path.join(save_path, f'seed-{seed}_epoch-0')
        torch.save(pytorch_model.state_dict(), initial_checkpoint_path)
        print(f"  Saved initial checkpoint at epoch 0")

    # Train model
    if num_epochs > 0:
        trainer.fit(model=lightning_model, datamodule=data_module)

    # Save final checkpoint if not already in checkpoint_epochs
    if num_epochs not in checkpoint_epochs and num_epochs > 0:
        final_checkpoint_path = os.path.join(save_path, f'seed-{seed}_epoch-{num_epochs}')
        torch.save(pytorch_model.state_dict(), final_checkpoint_path)
        print(f"  Saved final checkpoint at epoch {num_epochs}")

    # Clean up memory
    del pytorch_model
    del lightning_model
    del trainer
    torch.cuda.empty_cache()

    return save_path


def train_multiple_seeds_with_checkpoints(
    config,
    num_seeds=5,
    start_seed=0,
    checkpoint_epochs=None
):
    """
    Train multiple models with different seeds, saving checkpoints for each.

    Args:
        config: Configuration dictionary with training parameters
        num_seeds: Number of models to train
        start_seed: Starting seed value
        checkpoint_epochs: List of epochs to save checkpoints at
    """
    print(f"Training {num_seeds} models with checkpoints")
    print(f"Checkpoint epochs: {checkpoint_epochs or CHECKPOINT_EPOCHS}")

    for seed in range(start_seed, start_seed + num_seeds):
        print(f"\nTraining model with seed {seed}")

        # Vary hidden dimension if necessary
        current_config = config.copy()
        if current_config.get('varying_dim_bounds') is not None:
            random_dimension = np.random.randint(*current_config['varying_dim_bounds'])
            current_config['hidden_dim'] = [random_dimension] * len(current_config['hidden_dim'])

        save_path = train_with_checkpoints(
            model_class_str=current_config['model_class_str'],
            dataset_class_str=current_config['dataset_class_str'],
            batch_size=current_config['batch_size'],
            num_epochs=current_config['num_epochs'],
            learning_rate=current_config['learning_rate'],
            num_workers=current_config['num_workers'],
            num_classes=current_config['num_classes'],
            hidden_dim=current_config['hidden_dim'],
            seed=seed,
            data_dir='./data',
            models_dir=current_config.get('models_dir', 'saved_models_checkpoints/'),
            checkpoint_epochs=checkpoint_epochs,
            varying_dim_bounds=current_config.get('varying_dim_bounds')
        )

        print(f"  Checkpoints saved to: {save_path}")


if __name__ == '__main__':
    # Example configuration for training dynamics experiment
    CONFIG = {
        'model_class_str': 'fully_connected_dropout',
        'dataset_class_str': 'mnist',
        'batch_size': 256,
        'num_epochs': 100,  # Train for longer to see full dynamics
        'learning_rate': 0.001,
        'num_workers': 4,
        'num_classes': 10,
        'hidden_dim': [50, 50],
        'varying_dim_bounds': None,
        'models_dir': 'saved_models_checkpoints/'
    }

    # Train 5 models with checkpoints
    train_multiple_seeds_with_checkpoints(
        config=CONFIG,
        num_seeds=5,
        start_seed=0,
        checkpoint_epochs=CHECKPOINT_EPOCHS
    )
