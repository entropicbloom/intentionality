"""
Dataset for loading checkpointed models at specific epochs.
Used for the Training Dynamics of Intentionality experiment.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class CheckpointedLastLayerDataset(Dataset):
    """
    Dataset for extracting weights from checkpointed models at specific epochs.

    This dataset loads models saved at different training checkpoints,
    allowing us to evaluate decoder performance across training dynamics.
    """
    def __init__(self, dataset_path, layer_idx, epoch, transpose_weights=False,
                 preprocessing=None, use_neurons=None, use_target_similarity_only=False):
        """
        Initialize the checkpointed dataset.

        Args:
            dataset_path (str): Path to directory containing checkpointed models
            layer_idx (int): Index of the layer to extract weights from
            epoch (int): Specific epoch checkpoint to load
            transpose_weights (bool): Whether to transpose the weight matrix
            preprocessing (str, optional): Preprocessing method to apply
            use_neurons (list, optional): List of specific neuron indices to use
            use_target_similarity_only (bool): Whether to use only target similarity vector
        """
        self.dataset_path = dataset_path
        self.layer = f'layers.{layer_idx}.weight'
        self.epoch = epoch
        self.transpose_weights = transpose_weights
        self.preprocessing = preprocessing
        self.use_neurons = use_neurons
        self.use_target_similarity_only = use_target_similarity_only

        # Find all checkpoint files for this epoch
        self.checkpoint_files = self._find_checkpoint_files()

        if len(self.checkpoint_files) == 0:
            raise ValueError(f"No checkpoint files found for epoch {epoch} in {dataset_path}")

        # Load a sample checkpoint to get dimensions
        sample_checkpoint = torch.load(os.path.join(self.dataset_path, self.checkpoint_files[0]))
        if not transpose_weights:
            self.num_classes = sample_checkpoint[self.layer].shape[0]
        else:
            self.num_classes = sample_checkpoint[self.layer].shape[1]

        # If we're using specific neurons, update the effective number of classes
        if self.use_neurons is not None:
            self.effective_num_classes = len(self.use_neurons)
        else:
            self.effective_num_classes = self.num_classes

    def _find_checkpoint_files(self):
        """Find all checkpoint files for the specified epoch."""
        checkpoint_files = []
        for filename in os.listdir(self.dataset_path):
            if filename.startswith('seed-') and f'_epoch-{self.epoch}' in filename:
                checkpoint_files.append(filename)

        # Sort by seed number for consistency
        checkpoint_files.sort(key=lambda x: int(x.split('seed-')[1].split('_')[0]))
        return checkpoint_files

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.checkpoint_files) * self.effective_num_classes

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve

        Returns:
            tuple: (weights, class_index)
        """
        # Get model and class indices
        model_idx = idx // self.effective_num_classes
        neuron_idx = idx % self.effective_num_classes

        # Map to actual neuron index if using subset
        if self.use_neurons is not None:
            class_idx = self.use_neurons[neuron_idx]
        else:
            class_idx = neuron_idx

        # Load checkpoint
        checkpoint_file = self.checkpoint_files[model_idx]
        model = torch.load(os.path.join(self.dataset_path, checkpoint_file))
        weights = model[self.layer].to('cpu')

        if self.transpose_weights:
            weights = weights.T

        # Filter to only use specified neurons if requested
        if self.use_neurons is not None:
            weights = weights[self.use_neurons, :]
            class_idx_in_filtered = self.use_neurons.index(class_idx)
        else:
            class_idx_in_filtered = class_idx

        # Shuffle rows of weight matrix (same as LastLayerDataset)
        tmp = weights[class_idx_in_filtered].clone()
        weights[class_idx_in_filtered] = weights[0]
        weights[0] = tmp

        if weights.shape[0] > 1:
            shuffle_indices = torch.randperm(weights.shape[0] - 1)
            weights[1:,:] = weights[1:,:][shuffle_indices]

        # Apply preprocessing
        if self.preprocessing == 'multiply_transpose':
            weights_norm = weights / torch.norm(weights, dim=1, keepdim=True)
            sim_matrix = weights_norm @ weights_norm.T
            if self.use_target_similarity_only:
                weights = sim_matrix[0:1, :]
            else:
                weights = sim_matrix
        elif self.preprocessing == 'dim_reduction':
            U, _, _ = torch.pca_lowrank(weights.T, q=self.num_classes, center=True)
            weights = weights @ U
            weights = weights[:,torch.randperm(weights.shape[1])]

        return weights, torch.Tensor([class_idx_in_filtered])


class CheckpointedLastLayerDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the CheckpointedLastLayerDataset.
    """
    def __init__(self, dataset_path, layer_idx, epoch, input_dim, batch_size, num_workers,
                 transpose_weights=False, preprocessing=None, use_neurons=None,
                 use_target_similarity_only=False):
        """
        Initialize the checkpointed data module.

        Args:
            dataset_path (str): Path to directory containing checkpointed models
            layer_idx (int): Index of the layer to extract weights from
            epoch (int): Specific epoch checkpoint to load
            input_dim (int): Input dimension for the model
            batch_size (int): Batch size for DataLoaders
            num_workers (int): Number of workers for DataLoaders
            transpose_weights (bool): Whether to transpose the weight matrix
            preprocessing (str, optional): Preprocessing method to apply
            use_neurons (list, optional): List of specific neuron indices to use
            use_target_similarity_only (bool): Whether to use only target similarity vector
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.layer_idx = layer_idx
        self.epoch = epoch
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transpose_weights = transpose_weights
        self.preprocessing = preprocessing
        self.use_neurons = use_neurons
        self.use_target_similarity_only = use_target_similarity_only

    def prepare_data(self):
        return

    def setup(self, stage=None):
        """Set up the dataset and create train/validation splits."""
        dataset = CheckpointedLastLayerDataset(
            self.dataset_path,
            self.layer_idx,
            self.epoch,
            transpose_weights=self.transpose_weights,
            preprocessing=self.preprocessing,
            use_neurons=self.use_neurons,
            use_target_similarity_only=self.use_target_similarity_only
        )

        # 80/20 train/validation split
        train_size = int(len(dataset) * 0.8)
        self.train = torch.utils.data.Subset(dataset, range(train_size))
        self.valid = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
        self.test = None

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
        )
