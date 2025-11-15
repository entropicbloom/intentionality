import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader

from underlying.utils import get_dir_path
from decoder.config import get_underlying_path

class DatasetClassificationDataset(Dataset):
    """
    Dataset for classifying whether a neural network was trained on MNIST or Fashion-MNIST
    based on cosine similarities of output neurons.
    
    This dataset extracts output layer weights from trained models and computes cosine
    similarities between output neurons as features for binary classification.
    """
    
    def __init__(self, train_samples, valid_samples, config, split='train'):
        """
        Initialize the dataset.
        
        Args:
            train_samples (int): Number of training samples
            valid_samples (int): Number of validation samples  
            config (dict): Configuration dictionary
            split (str): 'train' or 'valid'
        """
        self.config = config
        self.split = split
        self.train_samples = train_samples
        self.valid_samples = valid_samples
        
        # Get paths for both MNIST and Fashion-MNIST models
        self.mnist_path = get_underlying_path(get_dir_path(
            model_class_str=config['model_class_str'],
            dataset_class_str='mnist',
            num_epochs=0 if config.get('untrained', False) else 2,
            hidden_dim=config.get('hidden_dim', [50, 50]),
            varying_dim=config.get('varying_dim', False),
            models_dir=config.get('models_dir', 'saved_models/')
        ))

        self.fashionmnist_path = get_underlying_path(get_dir_path(
            model_class_str=config['model_class_str'],
            dataset_class_str='fashionmnist',
            num_epochs=0 if config.get('untrained', False) else 2,
            hidden_dim=config.get('hidden_dim', [50, 50]),
            varying_dim=config.get('varying_dim', False),
            models_dir=config.get('models_dir', 'saved_models/')
        ))

        # Calculate layer index based on number of hidden layers
        # hidden_dim=[50, 50] -> 2 hidden layers -> output at index 2
        # hidden_dim=[100] -> 1 hidden layer -> output at index 1
        hidden_dim = config.get('hidden_dim', [50, 50])
        layer_idx = len(hidden_dim)
        self.layer = f'layers.{layer_idx}.weight'
        
        # Set up train/valid split
        if split == 'train':
            self.num_models_per_dataset = train_samples // 2  # Half MNIST, half Fashion-MNIST
        else:
            self.num_models_per_dataset = valid_samples // 2
            
    def __len__(self):
        """Return the total number of samples."""
        return self.num_models_per_dataset * 2  # Two datasets
        
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (cosine_similarities, dataset_label)
                - cosine_similarities: Tensor of cosine similarities between output neurons
                - dataset_label: 0 for MNIST, 1 for Fashion-MNIST
        """
        # Determine which dataset and model
        if idx < self.num_models_per_dataset:
            # MNIST
            dataset_label = 0
            model_idx = idx
            dataset_path = self.mnist_path
        else:
            # Fashion-MNIST
            dataset_label = 1
            model_idx = idx - self.num_models_per_dataset
            dataset_path = self.fashionmnist_path
            
        # Offset model indices for validation split
        if self.split == 'valid':
            model_idx += self.train_samples // 2  # Start from where training ends
            
        # Load model weights
        model = torch.load(dataset_path + f'seed-{model_idx}')
        weights = model[self.layer].to('cpu')  # Shape: [10, hidden_dim] for output layer
        
        # Randomly permute the neuron order to avoid any positional bias
        # This follows the same pattern as last_layer.py with preprocessing='multiply_transpose'
        perm_indices = torch.randperm(weights.shape[0])
        weights = weights[perm_indices, :]
        
        # Compute cosine similarities between output neurons (multiply_transpose preprocessing)
        # Normalize weights first
        weights_norm = weights / torch.norm(weights, dim=1, keepdim=True)
        # Compute cosine similarity matrix
        sim_matrix = weights_norm @ weights_norm.T  # Shape: [10, 10]
        
        # Return the full cosine similarity matrix (no upper triangular extraction)
        # This matches the behavior of last_layer.py with preprocessing='multiply_transpose'
        # Keep the 2D matrix structure for self-attention decoder
        cosine_similarities = sim_matrix  # Shape: [10, 10]
        
        return cosine_similarities, torch.tensor(dataset_label, dtype=torch.long)

class DatasetClassificationDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the DatasetClassificationDataset.
    """
    
    def __init__(self, batch_size, num_workers, train_samples, valid_samples, config):
        """
        Initialize the DataModule.
        
        Args:
            batch_size (int): Batch size for data loaders
            num_workers (int): Number of workers for data loading
            train_samples (int): Number of training samples
            valid_samples (int): Number of validation samples
            config (dict): Configuration dictionary
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_samples = train_samples
        self.valid_samples = valid_samples
        self.config = config
        
    def setup(self, stage=None):
        """Set up the datasets."""
        self.train_dataset = DatasetClassificationDataset(
            train_samples=self.train_samples,
            valid_samples=self.valid_samples,
            config=self.config,
            split='train'
        )
        
        self.valid_dataset = DatasetClassificationDataset(
            train_samples=self.train_samples,
            valid_samples=self.valid_samples,
            config=self.config,
            split='valid'
        )
        
    def train_dataloader(self):
        """Return the training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0
        )
        
    def val_dataloader(self):
        """Return the validation data loader.""" 
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0
        )