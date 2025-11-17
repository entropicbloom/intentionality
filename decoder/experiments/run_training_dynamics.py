"""
Main experiment runner for Training Dynamics of Intentionality.

This experiment investigates when intentionality emerges during training by:
1. Training underlying models and saving checkpoints at specific epochs
2. Training decoders on each checkpoint to measure decoder accuracy
3. Tracking the underlying model's task accuracy at each checkpoint
4. Computing entropy reduction (ARS) over training

Usage:
    python -m decoder.experiments.run_training_dynamics
"""
import os
import sys
import csv
import torch
import pytorch_lightning as pl
from decoder.config import config as base_config
from decoder.setup.training_dynamics import run_training_dynamics_experiment, save_results_to_csv
from underlying.train_with_checkpoints import train_multiple_seeds_with_checkpoints

# Add underlying directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../underlying'))

from underlying.datasets.MNIST import MNISTDataModule
from underlying.datasets.FashionMNIST import FashionMNISTDataModule
from underlying.pytorch_models.fully_connected import FullyConnected, FullyConnectedDropout
from underlying.lightning_model import LightningModel
from underlying.utils import get_dir_path


CHECKPOINT_EPOCHS = [0, 1, 2, 3, 5, 10, 20, 50, 100]

DATASET_MAP = {
    'mnist': MNISTDataModule,
    'fashionmnist': FashionMNISTDataModule
}

MODEL_MAP = {
    'fully_connected': FullyConnected,
    'fully_connected_dropout': FullyConnectedDropout
}


def evaluate_underlying_model_accuracy(checkpoint_path, dataset_class_str, batch_size=256, num_workers=0):
    """
    Evaluate the underlying model's task accuracy on a checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file
        dataset_class_str (str): Dataset identifier
        batch_size (int): Batch size for evaluation
        num_workers (int): Number of data loading workers

    Returns:
        float: Validation accuracy of the underlying model
    """
    # Load checkpoint
    state_dict = torch.load(checkpoint_path)

    # Get dataset
    dataset_class = DATASET_MAP[dataset_class_str]
    data_module = dataset_class(batch_size, num_workers, data_path='./data')
    data_module.setup()

    # Reconstruct model architecture from state dict
    # Infer hidden dimensions from weight shapes
    hidden_dims = []
    layer_idx = 0
    while f'layers.{layer_idx}.weight' in state_dict:
        if layer_idx == 0:
            input_dim = state_dict[f'layers.{layer_idx}.weight'].shape[1]
        hidden_dims.append(state_dict[f'layers.{layer_idx}.weight'].shape[0])
        layer_idx += 1

    # Last layer is output, so remove it from hidden_dims
    output_dim = hidden_dims.pop()
    num_classes = output_dim

    # Determine if dropout model based on checkpoint path
    if 'dropout' in checkpoint_path:
        model_class = FullyConnectedDropout
    else:
        model_class = FullyConnected

    # Create model
    pytorch_model = model_class(
        num_classes=num_classes,
        input_dim=data_module.input_dim,
        hidden_dim=hidden_dims
    )

    # Load weights
    pytorch_model.load_state_dict(state_dict)

    # Wrap in Lightning module
    lightning_model = LightningModel(pytorch_model, learning_rate=0.001, num_classes=num_classes)

    # Evaluate
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    results = trainer.validate(model=lightning_model, dataloaders=data_module.val_dataloader(), verbose=False)

    return float(results[0]['valid_acc'])


def compute_task_accuracies(config, checkpoint_epochs, num_seeds):
    """
    Compute task accuracy for all checkpoints.

    Args:
        config (dict): Configuration dictionary
        checkpoint_epochs (list): List of epochs to evaluate
        num_seeds (int): Number of model seeds

    Returns:
        dict: Dictionary mapping (epoch, seed) -> task_accuracy
    """
    dataset_path = '../underlying/' + get_dir_path(
        model_class_str=config['model_class_str'],
        dataset_class_str=config['dataset_class_str'],
        num_epochs=config['underlying_num_epochs'],
        hidden_dim=config['hidden_dim'],
        varying_dim=config['varying_dim'],
        models_dir=config['models_dir']
    )

    task_accuracies = {}

    print("\n  Computing task accuracies for underlying models...")
    for seed in range(num_seeds):
        for epoch in checkpoint_epochs:
            checkpoint_file = f'seed-{seed}_epoch-{epoch}'
            checkpoint_path = os.path.join(dataset_path, checkpoint_file)

            if os.path.exists(checkpoint_path):
                print(f"    Seed {seed}, Epoch {epoch}")
                try:
                    task_acc = evaluate_underlying_model_accuracy(
                        checkpoint_path,
                        config['dataset_class_str']
                    )
                    task_accuracies[(epoch, seed)] = task_acc
                except Exception as e:
                    print(f"      Error evaluating: {e}")
                    task_accuracies[(epoch, seed)] = 0.0

    return task_accuracies


def run_full_training_dynamics_experiment(
    num_underlying_seeds=5,
    num_decoder_seeds=5,
    num_neurons=10,
    underlying_config=None,
    decoder_config=None,
    checkpoint_epochs=None,
    train_decoder_per_epoch=True
):
    """
    Run the complete training dynamics experiment.

    Args:
        num_underlying_seeds (int): Number of underlying model seeds to train
        num_decoder_seeds (int): Number of decoder seeds to train per checkpoint
        num_neurons (int): Number of neurons to use for decoding
        underlying_config (dict): Configuration for underlying models
        decoder_config (dict): Configuration for decoder models
        checkpoint_epochs (list): List of epochs to save/evaluate checkpoints
        train_decoder_per_epoch (bool): Whether to train a new decoder for each epoch

    Returns:
        str: Path to results CSV file
    """
    if checkpoint_epochs is None:
        checkpoint_epochs = CHECKPOINT_EPOCHS

    if underlying_config is None:
        underlying_config = {
            'model_class_str': 'fully_connected_dropout',
            'dataset_class_str': 'mnist',
            'batch_size': 256,
            'num_epochs': 100,
            'learning_rate': 0.001,
            'num_workers': 4,
            'num_classes': 10,
            'hidden_dim': [50, 50],
            'varying_dim_bounds': None,
            'models_dir': 'saved_models_checkpoints/'
        }

    if decoder_config is None:
        decoder_config = base_config.copy()
        decoder_config['model_class_str'] = underlying_config['model_class_str']
        decoder_config['dataset_class_str'] = underlying_config['dataset_class_str']
        decoder_config['hidden_dim'] = underlying_config['hidden_dim']
        decoder_config['varying_dim'] = False
        decoder_config['models_dir'] = underlying_config['models_dir']
        decoder_config['underlying_num_epochs'] = underlying_config['num_epochs']
        decoder_config['decoder_class'] = 'TransformerDecoder'
        decoder_config['preprocessing'] = 'multiply_transpose'

    # Step 1: Train underlying models with checkpoints
    print("=" * 80)
    print("STEP 1: Training underlying models with checkpoints")
    print("=" * 80)
    train_multiple_seeds_with_checkpoints(
        config=underlying_config,
        num_seeds=num_underlying_seeds,
        start_seed=0,
        checkpoint_epochs=checkpoint_epochs
    )

    # Step 2: Compute task accuracies for all checkpoints
    print("\n" + "=" * 80)
    print("STEP 2: Computing task accuracies")
    print("=" * 80)
    task_accuracies = compute_task_accuracies(
        config=decoder_config,
        checkpoint_epochs=checkpoint_epochs,
        num_seeds=num_underlying_seeds
    )

    # Step 3: Train decoders and evaluate on each checkpoint
    print("\n" + "=" * 80)
    print("STEP 3: Training decoders and evaluating on checkpoints")
    print("=" * 80)
    all_results = []

    for decoder_seed in range(num_decoder_seeds):
        print(f"\nDecoder seed {decoder_seed}")
        results = run_training_dynamics_experiment(
            seed=decoder_seed,
            num_neurons=num_neurons,
            config=decoder_config,
            checkpoint_epochs=checkpoint_epochs,
            train_decoder_per_epoch=train_decoder_per_epoch,
            save_dir='logs/training_dynamics'
        )

        # Add decoder seed and task accuracy to results
        for result in results:
            result['decoder_seed'] = decoder_seed
            # Average task accuracy across all underlying seeds for this epoch
            epoch = result['epoch']
            task_accs = [task_accuracies.get((epoch, s), 0.0) for s in range(num_underlying_seeds)]
            result['task_acc_mean'] = sum(task_accs) / len(task_accs) if len(task_accs) > 0 else 0.0
            result['num_neurons'] = num_neurons

        all_results.extend(results)

    # Step 4: Save results
    print("\n" + "=" * 80)
    print("STEP 4: Saving results")
    print("=" * 80)
    results_dir = 'data/training-dynamics'
    os.makedirs(results_dir, exist_ok=True)

    config_str = f"{decoder_config['model_class_str']}_{decoder_config['dataset_class_str']}_n{num_neurons}"
    results_file = os.path.join(results_dir, f'training_dynamics_{config_str}.csv')

    save_results_to_csv(all_results, results_file)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {results_file}")

    return results_file


if __name__ == '__main__':
    # Run with default parameters (small scale for testing)
    run_full_training_dynamics_experiment(
        num_underlying_seeds=5,
        num_decoder_seeds=3,
        num_neurons=10,
        train_decoder_per_epoch=True
    )
