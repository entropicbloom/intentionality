"""
Setup and training functions for the Training Dynamics of Intentionality experiment.
"""
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import os
import csv

from decoder.underlying_datasets.checkpointed_last_layer import CheckpointedLastLayerDataModule
from decoder.lightning_model import LightningClassificationModel
from decoder.models import decoder_dict
from underlying.utils import get_dir_path


def evaluate_decoder_on_checkpoint(
    seed,
    epoch,
    num_neurons,
    config,
    trained_decoder=None,
    save_dir=None
):
    """
    Evaluate a decoder on a specific checkpoint epoch.

    Args:
        seed (int): Random seed for decoder training
        epoch (int): Checkpoint epoch to evaluate
        num_neurons (int): Number of neurons to use
        config (dict): Configuration dictionary
        trained_decoder (LightningClassificationModel, optional): Pre-trained decoder to use
        save_dir (str, optional): Directory to save results

    Returns:
        dict: Results containing epoch, train_acc, valid_acc
    """
    torch.manual_seed(seed)

    # Get dataset path for underlying models
    dataset_path = '../underlying/' + get_dir_path(
        model_class_str=config['model_class_str'],
        dataset_class_str=config['dataset_class_str'],
        num_epochs=config['underlying_num_epochs'],
        hidden_dim=config['hidden_dim'],
        varying_dim=config['varying_dim'],
        models_dir=config['models_dir']
    )

    # Create list of neurons to use
    use_neurons = list(range(num_neurons))

    # Create data module for this checkpoint epoch
    data_module = CheckpointedLastLayerDataModule(
        dataset_path,
        layer_idx=2,
        epoch=epoch,
        input_dim=50,
        batch_size=64,
        num_workers=0,
        transpose_weights=False,
        preprocessing=config.get('preprocessing', 'multiply_transpose'),
        use_neurons=use_neurons,
        use_target_similarity_only=config.get('use_target_similarity_only', False),
    )

    # Setup data
    data_module.setup()

    # If no pre-trained decoder is provided, train a new one
    if trained_decoder is None:
        # Initialize new decoder model
        pytorch_model = decoder_dict[config['decoder_class']](
            dim_input=num_neurons,
            num_outputs=1,
            dim_output=10,
            num_inds=16,
            dim_hidden=64,
            num_heads=4,
            ln=False
        )
        lightning_model = LightningClassificationModel(
            pytorch_model,
            learning_rate=0.001,
            num_classes=10
        )

        # Train decoder on this checkpoint
        callbacks = [ModelCheckpoint(save_top_k=1, mode="max", monitor="valid_acc")]
        trainer = pl.Trainer(
            max_epochs=200,
            callbacks=callbacks,
            accelerator="auto",
            devices="auto",
            deterministic=False,
            log_every_n_steps=10,
            logger=CSVLogger(save_dir=save_dir or 'logs/', name=f"training-dynamics-epoch-{epoch}"),
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        # Train the decoder
        trainer.fit(model=lightning_model, datamodule=data_module)

        # Get final metrics
        train_acc = trainer.callback_metrics.get('train_acc', 0.0)
        valid_acc = trainer.callback_metrics.get('valid_acc', 0.0)
    else:
        # Use pre-trained decoder for evaluation
        trainer = pl.Trainer(
            accelerator="auto",
            devices="auto",
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        # Evaluate on both train and validation sets
        train_results = trainer.validate(
            model=trained_decoder,
            dataloaders=data_module.train_dataloader(),
            verbose=False
        )
        valid_results = trainer.validate(
            model=trained_decoder,
            dataloaders=data_module.val_dataloader(),
            verbose=False
        )

        train_acc = train_results[0].get('valid_acc', 0.0)  # Metric name is 'valid_acc' even for train
        valid_acc = valid_results[0].get('valid_acc', 0.0)

    return {
        'epoch': epoch,
        'train_acc': float(train_acc),
        'valid_acc': float(valid_acc)
    }


def run_training_dynamics_experiment(
    seed,
    num_neurons,
    config,
    checkpoint_epochs=None,
    train_decoder_per_epoch=True,
    save_dir=None
):
    """
    Run the full training dynamics experiment for a single decoder seed.

    Args:
        seed (int): Random seed for decoder training
        num_neurons (int): Number of neurons to use
        config (dict): Configuration dictionary
        checkpoint_epochs (list, optional): List of epochs to evaluate
        train_decoder_per_epoch (bool): If True, train a new decoder for each epoch.
                                       If False, train one decoder on final epoch and evaluate on all.
        save_dir (str, optional): Directory to save results

    Returns:
        list: List of result dictionaries, one per checkpoint epoch
    """
    if checkpoint_epochs is None:
        checkpoint_epochs = [0, 1, 2, 3, 5, 10, 20, 50, 100]

    # Filter epochs that actually exist
    dataset_path = '../underlying/' + get_dir_path(
        model_class_str=config['model_class_str'],
        dataset_class_str=config['dataset_class_str'],
        num_epochs=config['underlying_num_epochs'],
        hidden_dim=config['hidden_dim'],
        varying_dim=config['varying_dim'],
        models_dir=config['models_dir']
    )

    available_epochs = []
    for epoch in checkpoint_epochs:
        # Check if checkpoint files exist for this epoch
        checkpoint_files = [f for f in os.listdir(dataset_path)
                           if f.startswith('seed-') and f'_epoch-{epoch}' in f]
        if len(checkpoint_files) > 0:
            available_epochs.append(epoch)

    print(f"  Available epochs: {available_epochs}")

    results = []

    if train_decoder_per_epoch:
        # Train a new decoder for each epoch's checkpoint
        for epoch in available_epochs:
            print(f"    Evaluating epoch {epoch}")
            result = evaluate_decoder_on_checkpoint(
                seed=seed,
                epoch=epoch,
                num_neurons=num_neurons,
                config=config,
                trained_decoder=None,
                save_dir=save_dir
            )
            results.append(result)
    else:
        # Train one decoder on final epoch, then evaluate on all epochs
        # (This tests whether a decoder trained on fully trained networks can decode earlier states)
        final_epoch = max(available_epochs)
        print(f"    Training decoder on final epoch {final_epoch}")

        # Train decoder on final epoch
        torch.manual_seed(seed)
        use_neurons = list(range(num_neurons))

        # Get final epoch data module
        data_module_final = CheckpointedLastLayerDataModule(
            dataset_path,
            layer_idx=2,
            epoch=final_epoch,
            input_dim=50,
            batch_size=64,
            num_workers=0,
            transpose_weights=False,
            preprocessing=config.get('preprocessing', 'multiply_transpose'),
            use_neurons=use_neurons,
            use_target_similarity_only=config.get('use_target_similarity_only', False),
        )

        # Initialize and train decoder
        pytorch_model = decoder_dict[config['decoder_class']](
            dim_input=num_neurons,
            num_outputs=1,
            dim_output=10,
            num_inds=16,
            dim_hidden=64,
            num_heads=4,
            ln=False
        )
        trained_decoder = LightningClassificationModel(
            pytorch_model,
            learning_rate=0.001,
            num_classes=10
        )

        callbacks = [ModelCheckpoint(save_top_k=1, mode="max", monitor="valid_acc")]
        trainer = pl.Trainer(
            max_epochs=200,
            callbacks=callbacks,
            accelerator="auto",
            devices="auto",
            deterministic=False,
            log_every_n_steps=10,
            logger=CSVLogger(save_dir=save_dir or 'logs/', name=f"training-dynamics-final"),
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        trainer.fit(model=trained_decoder, datamodule=data_module_final)

        # Now evaluate this trained decoder on all epochs
        for epoch in available_epochs:
            print(f"    Evaluating epoch {epoch} with final-trained decoder")
            result = evaluate_decoder_on_checkpoint(
                seed=seed,
                epoch=epoch,
                num_neurons=num_neurons,
                config=config,
                trained_decoder=trained_decoder,
                save_dir=save_dir
            )
            results.append(result)

    return results


def save_results_to_csv(results, filepath):
    """
    Save experiment results to CSV file.

    Args:
        results (list): List of result dictionaries
        filepath (str): Path to save CSV file
    """
    if len(results) == 0:
        return

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Get all keys from results
    fieldnames = list(results[0].keys())

    # Write to CSV
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {filepath}")
