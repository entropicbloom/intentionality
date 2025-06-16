import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import math

class LastLayerDataset(Dataset):
    """
    Dataset for extracting neural network weights from trained models for decoding.
    
    This dataset extracts weights from a specific layer of multiple trained models,
    allowing for experiments on weight interpretability. It supports using only a 
    subset of neurons through the use_neurons parameter.
    
    Attributes:
        dataset_path: Path to directory containing saved models
        layer: Name of the layer to extract weights from
        transpose_weights: Whether to transpose the weight matrix
        preprocessing: Optional preprocessing method to apply to weights
        use_neurons: Optional list of specific neuron indices to use
        num_classes: Total number of classes/neurons in the original weight matrix
        effective_num_classes: Number of classes/neurons after filtering (if applicable)
        use_target_similarity_only: Whether to use only the similarity vector of the target neuron when preprocessing is 'multiply_transpose'.
    """
    def __init__(self, dataset_path, layer_idx, transpose_weights=False, preprocessing=None, use_neurons=None,
                use_target_similarity_only: bool = False):
        """
        Initialize the dataset.
        
        Args:
            dataset_path (str): Path to directory containing saved models
            layer_idx (int): Index of the layer to extract weights from
            transpose_weights (bool): Whether to transpose the weight matrix
            preprocessing (str, optional): Preprocessing method to apply ('multiply_transpose' or 'dim_reduction')
            use_neurons (list, optional): List of specific neuron indices to use. If provided, only these
                                         neurons will be included in the dataset.
            use_target_similarity_only (bool, optional): If True and preprocessing is 'multiply_transpose', 
                                                     only the target neuron's similarity vector is returned. Defaults to False.
        """
        self.dataset_path = dataset_path
        self.layer = f'layers.{layer_idx}.weight'
        self.transpose_weights = transpose_weights
        self.preprocessing = preprocessing
        self.use_neurons = use_neurons
        self.use_target_similarity_only = use_target_similarity_only

        if not transpose_weights:
            self.num_classes = torch.load(self.dataset_path + f'seed-{0}')[self.layer].shape[0]
        else:
            self.num_classes = torch.load(self.dataset_path + f'seed-{0}')[self.layer].shape[1]
            
        # If we're using specific neurons, update the effective number of classes
        if self.use_neurons is not None:
            self.effective_num_classes = len(self.use_neurons)
        else:
            self.effective_num_classes = self.num_classes

    def __len__(self):
        """
        Return the length of the dataset.
        
        Returns:
            int: Number of samples in the dataset (number of models × effective number of classes)
        """
        # Only count files that match the expected model filename pattern
        num_models = sum(1 for f in os.listdir(self.dataset_path) if f.startswith('seed-') and f[5:].isdigit())
        return num_models * self.effective_num_classes  # Use effective number of classes

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        When use_neurons is specified, this method:
        1. Maps the dataset index to the appropriate model and neuron index
        2. Filters the weight matrix to only include specified neurons
        3. Maps the class index to its position in the filtered weights
        4. Shuffles the rows of the weight matrix to ensure the model must learn
           to recognize each neuron's weight pattern
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (weights, class_index)
                - weights: Tensor of weight values 
                - class_index: Tensor containing the class index (position in filtered weights)
        """
        # get model and class indices based on effective number of classes
        model_idx = idx // self.effective_num_classes
        neuron_idx = idx % self.effective_num_classes
        
        # If we're using specific neurons, map the neuron_idx to the actual neuron index
        if self.use_neurons is not None:
            class_idx = self.use_neurons[neuron_idx]
        else:
            class_idx = neuron_idx

        # load relevant data
        model = torch.load(self.dataset_path + f'seed-{model_idx}')
        weights = model[self.layer].to('cpu')
        if self.transpose_weights:
            weights = weights.T
            
        # Filter to only use specified neurons if requested
        if self.use_neurons is not None:
            weights = weights[self.use_neurons, :]
            # Now class_idx needs to be mapped to its position in the filtered weights
            class_idx_in_filtered = self.use_neurons.index(class_idx)
        else:
            class_idx_in_filtered = class_idx

        # shuffle rows of weight matrix
        tmp = weights[class_idx_in_filtered].clone()
        weights[class_idx_in_filtered] = weights[0]
        weights[0] = tmp

        if weights.shape[0] > 1:  # Only shuffle if there's more than one row
            # Get indices excluding the first element (which now contains our target class)
            shuffle_indices = torch.randperm(weights.shape[0] - 1)
            weights[1:,:] = weights[1:,:][shuffle_indices]

        # apply preprocessing 
        if self.preprocessing == 'multiply_transpose':
            # Normalize weights first
            weights_norm = weights / torch.norm(weights, dim=1, keepdim=True)
            # Then compute cosine similarities
            sim_matrix = weights_norm @ weights_norm.T
            if self.use_target_similarity_only:
                weights = sim_matrix[0:1, :] # Return only the first row (target neuron similarity) as a 2D tensor
            else:
                weights = sim_matrix # Return the full similarity matrix

        elif self.preprocessing == 'dim_reduction':
            U, _, _ = torch.pca_lowrank(weights.T, q=self.num_classes, center=True)
            weights = weights @ U
            
            # permute weights columns
            weights = weights[:,torch.randperm(weights.shape[1])]

        return weights, torch.Tensor([class_idx_in_filtered])

class LastLayerDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the LastLayerDataset.
    
    This module handles dataset creation, splitting into train/validation sets,
    and creating appropriate DataLoaders.
    
    Attributes:
        dataset_path: Path to directory containing saved models
        layer_idx: Index of the layer to extract weights from
        input_dim: Input dimension for the model
        batch_size: Batch size for training/validation
        num_workers: Number of workers for data loading
        transpose_weights: Whether to transpose the weight matrix
        preprocessing: Optional preprocessing method
        use_neurons: Optional list of specific neuron indices to use
        use_target_similarity_only: Whether to use only the target neuron's similarity vector.
    """
    def __init__(self, dataset_path, layer_idx, input_dim, batch_size, num_workers, 
                 transpose_weights=False, preprocessing=None, use_neurons=None, 
                 use_target_similarity_only: bool = False):
        """
        Initialize the data module.
        
        Args:
            dataset_path (str): Path to directory containing saved models
            layer_idx (int): Index of the layer to extract weights from
            input_dim (int): Input dimension for the model
            batch_size (int): Batch size for DataLoaders
            num_workers (int): Number of workers for DataLoaders
            transpose_weights (bool): Whether to transpose the weight matrix
            preprocessing (str, optional): Preprocessing method to apply
            use_neurons (list, optional): List of specific neuron indices to use.
                                         If provided, only these neurons will be included.
            use_target_similarity_only (bool, optional): Passed to LastLayerDataset. Defaults to False.
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_dim = input_dim
        self.dataset_path = dataset_path
        self.layer_idx = layer_idx
        self.transpose_weights = transpose_weights
        self.preprocessing = preprocessing
        self.use_neurons = use_neurons
        self.use_target_similarity_only = use_target_similarity_only

    def prepare_data(self):
        return

    def setup(self, stage=None):
        """
        Set up the dataset and create train/validation splits.
        
        This method creates a LastLayerDataset instance with the specified parameters,
        including any neuron filtering requested. Then splits it into train (80%) and
        validation (20%) subsets.
        
        Args:
            stage (str, optional): Stage of setup ('fit', 'validate', 'test')
        """
        dataset = LastLayerDataset(
            self.dataset_path, 
            self.layer_idx, 
            transpose_weights=self.transpose_weights, 
            preprocessing=self.preprocessing,
            use_neurons=self.use_neurons,
            use_target_similarity_only=self.use_target_similarity_only
        )

        # Created using indices from 0 to train_size.
        self.train = torch.utils.data.Subset(dataset, range(int(len(dataset) * 0.8)))
        self.valid = torch.utils.data.Subset(dataset, range(int(len(dataset) * 0.8), int(len(dataset))))
        self.test = None

    def train_dataloader(self):
        train_loader = DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            dataset=self.valid,
            batch_size=self.batch_size,
            drop_last=True, # TODO: need to figure out why this is necessary
            shuffle=False,
            num_workers=self.num_workers,
        )
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            dataset=self.test,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return test_loader

class MixedHiddenDimsDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule that uses different datasets for training and validation.
    This allows training on one set of models (e.g., hidden_dim=[100]) and validating 
    on another set (e.g., hidden_dim=[50, 50]).
    """
    def __init__(self, train_dataset_path, valid_dataset_path, train_layer_idx, valid_layer_idx, 
                 input_dim, batch_size, num_workers, transpose_weights=False, preprocessing=None, 
                 use_neurons=None, use_target_similarity_only=False, 
                 train_samples=800, valid_samples=200):
        """
        Initialize the mixed data module.
        
        Args:
            train_dataset_path (str): Path to training dataset
            valid_dataset_path (str): Path to validation dataset  
            train_layer_idx (int): Index of the layer to extract weights from for training
            valid_layer_idx (int): Index of the layer to extract weights from for validation
            input_dim (int): Input dimension for the model
            batch_size (int): Batch size for DataLoaders
            num_workers (int): Number of workers for DataLoaders
            transpose_weights (bool): Whether to transpose the weight matrix
            preprocessing (str, optional): Preprocessing method to apply
            use_neurons (list, optional): List of specific neuron indices to use
            use_target_similarity_only (bool): Whether to use only target similarity vector
            train_samples (int): Number of training samples to use
            valid_samples (int): Number of validation samples to use
        """
        super().__init__()
        self.train_dataset_path = train_dataset_path
        self.valid_dataset_path = valid_dataset_path
        self.train_layer_idx = train_layer_idx
        self.valid_layer_idx = valid_layer_idx
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transpose_weights = transpose_weights
        self.preprocessing = preprocessing
        self.use_neurons = use_neurons
        self.use_target_similarity_only = use_target_similarity_only
        self.train_samples = train_samples
        self.valid_samples = valid_samples

    def prepare_data(self):
        return

    def setup(self, stage=None):
        """
        Set up separate datasets for training and validation with different hidden dimensions.
        """
        # Create training dataset with appropriate layer index
        train_dataset = LastLayerDataset(
            self.train_dataset_path,
            self.train_layer_idx,
            transpose_weights=self.transpose_weights,
            preprocessing=self.preprocessing,
            use_neurons=self.use_neurons,
            use_target_similarity_only=self.use_target_similarity_only
        )
        
        # Create validation dataset with appropriate layer index
        valid_dataset = LastLayerDataset(
            self.valid_dataset_path,
            self.valid_layer_idx,
            transpose_weights=self.transpose_weights,
            preprocessing=self.preprocessing,
            use_neurons=self.use_neurons,
            use_target_similarity_only=self.use_target_similarity_only
        )
        
        # Limit to specified number of samples
        # If using the same dataset for both train and valid, ensure no overlap
        if self.train_dataset_path == self.valid_dataset_path:
            total_samples = len(train_dataset)
            train_indices = list(range(min(self.train_samples, total_samples - self.valid_samples)))
            valid_indices = list(range(total_samples - self.valid_samples, total_samples))
        else:
            # Different datasets, use from beginning of each
            train_indices = list(range(min(self.train_samples, len(train_dataset))))
            valid_indices = list(range(min(self.valid_samples, len(valid_dataset))))
        
        self.train = torch.utils.data.Subset(train_dataset, train_indices)
        self.valid = torch.utils.data.Subset(valid_dataset, valid_indices)
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