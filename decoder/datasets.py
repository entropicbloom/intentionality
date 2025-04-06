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
    """
    def __init__(self, dataset_path, layer_idx, transpose_weights=False, preprocessing=None, use_neurons=None):
        """
        Initialize the dataset.
        
        Args:
            dataset_path (str): Path to directory containing saved models
            layer_idx (int): Index of the layer to extract weights from
            transpose_weights (bool): Whether to transpose the weight matrix
            preprocessing (str, optional): Preprocessing method to apply ('multiply_transpose' or 'dim_reduction')
            use_neurons (list, optional): List of specific neuron indices to use. If provided, only these
                                         neurons will be included in the dataset.
        """
        self.dataset_path = dataset_path
        self.layer = f'layers.{layer_idx}.weight'
        self.transpose_weights = transpose_weights
        self.preprocessing = preprocessing
        self.use_neurons = use_neurons

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
            weights = weights_norm @ weights_norm.T

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
    """
    def __init__(self, dataset_path, layer_idx, input_dim, batch_size, num_workers, 
                 transpose_weights=False, preprocessing=None, use_neurons=None):
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
            use_neurons=self.use_neurons
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

def get_positional_encoding(pixel_idx, encoding_type):
    """Calculates positional encoding for a given pixel index."""
    i = pixel_idx // 28
    j = pixel_idx % 28
    if encoding_type == '2d_normalized':
        return torch.tensor([i / 27.0, j / 27.0], dtype=torch.float32)
    elif encoding_type == 'x_normalized':
        return torch.tensor([i / 27.0], dtype=torch.float32)
    elif encoding_type == 'y_normalized':
        return torch.tensor([j / 27.0], dtype=torch.float32)
    elif encoding_type == 'dist_center':
        dist = math.sqrt((i - 13.5)**2 + (j - 13.5)**2)
        # Normalize roughly to [0, 1] based on max possible distance (corner to center)
        max_dist = math.sqrt(13.5**2 + 13.5**2)
        return torch.tensor([dist / max_dist], dtype=torch.float32)
    else:
        raise ValueError(f"Unknown positional encoding type: {encoding_type}")

class FirstLayerDataset(Dataset):
    """
    Dataset for extracting the first layer weights from trained models 
    to decode input neuron positional information.
    
    Operates on columns (input neurons) of the first layer weight matrix.
    Applies cosine similarity preprocessing between columns.
    """
    def __init__(self, dataset_path, positional_encoding_type, layer_idx=0):
        self.dataset_path = dataset_path
        self.layer = f'layers.{layer_idx}.weight' # Should be layer 0 for first layer
        self.positional_encoding_type = positional_encoding_type
        self.num_input_neurons = 784 # MNIST specific 28*28

        # Determine number of models
        self.model_files = [f for f in os.listdir(self.dataset_path) if f.startswith('seed-') and f[5:].isdigit()]
        self.num_models = len(self.model_files)
        if self.num_models == 0:
            raise FileNotFoundError(f"No model files found in {self.dataset_path}")

    def __len__(self):
        """Return num_models * num_input_neurons"""
        return self.num_models * self.num_input_neurons

    def __getitem__(self, idx):
        model_idx = idx // self.num_input_neurons
        pixel_idx = idx % self.num_input_neurons
        
        # Load weights (H, 784)
        model_path = os.path.join(self.dataset_path, self.model_files[model_idx])
        model_state_dict = torch.load(model_path)
        weights = model_state_dict[self.layer].to('cpu') # Shape (Hidden_dim, 784)
        
        # Calculate label (positional encoding)
        label = get_positional_encoding(pixel_idx, self.positional_encoding_type)
        
        # --- Prepare input matrix by permuting columns --- 
        num_cols = weights.shape[1]
        if num_cols != self.num_input_neurons:
             raise ValueError(f"Expected {self.num_input_neurons} columns, but got {num_cols}")

        # Get target column
        target_col = weights[:, pixel_idx].clone()

        # Create permutation for other columns
        other_indices = [i for i in range(num_cols) if i != pixel_idx]
        permuted_other_indices = torch.randperm(len(other_indices)).tolist()
        shuffled_other_indices = [other_indices[i] for i in permuted_other_indices]
        
        # Create permuted weights matrix (H, 784)
        permuted_weights = torch.zeros_like(weights)
        permuted_weights[:, 0] = target_col
        permuted_weights[:, 1:] = weights[:, shuffled_other_indices]
        
        # --- Preprocessing: Column-wise Cosine Similarity --- 
        # Normalize columns
        norm_permuted_weights = permuted_weights / (torch.norm(permuted_weights, dim=0, keepdim=True) + 1e-8) # Add epsilon for stability
        # Compute similarity matrix X' = X_norm.T @ X_norm (784, 784)
        similarity_matrix = norm_permuted_weights.T @ norm_permuted_weights

        return similarity_matrix, label

class FirstLayerDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the FirstLayerDataset.
    Handles dataset creation, splitting, and DataLoaders for input pixel decoding.
    """
    def __init__(self, dataset_path, positional_encoding_type, batch_size, num_workers):
        super().__init__()
        self.dataset_path = dataset_path
        self.positional_encoding_type = positional_encoding_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        # Layer index is implicitly 0 for this DataModule

    def prepare_data(self):
        # Optional: download data, etc.
        pass

    def setup(self, stage=None):
        dataset = FirstLayerDataset(
            self.dataset_path, 
            positional_encoding_type=self.positional_encoding_type,
            layer_idx=0 
        )

        # Split data
        total_len = len(dataset)
        train_size = int(total_len * 0.8)
        val_size = total_len - train_size
        # Use random_split for better shuffling across models/pixels
        self.train, self.valid = random_split(dataset, [train_size, val_size])
        self.test = None # Or implement test split if needed

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True, # Often good for performance
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False # Usually keep all validation samples
        )

    def test_dataloader(self):
        if self.test is None:
             return None # Or raise error
        return DataLoader(
            dataset=self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )