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

class ViTOutputLayerDataset(Dataset):
    """
    Dataset for extracting the output layer weights from trained ViT models for decoding.
    
    This dataset extracts weights from the final classification head ('vit.head.weight') 
    of multiple trained ViT models, allowing for experiments on weight interpretability. 
    It supports using only a subset of neurons through the use of use_neurons parameter.
    
    Attributes:
        dataset_path: Path to directory containing saved model state_dict files (e.g., ending in .pt or .pth)
        transpose_weights: Whether to transpose the weight matrix (Likely False for standard ViT head)
        preprocessing: Optional preprocessing method to apply to weights
        use_neurons: Optional list of specific neuron indices to use
        num_classes: Total number of classes/neurons in the original weight matrix
        effective_num_classes: Number of classes/neurons after filtering (if applicable)
        use_target_similarity_only: Whether to use only the similarity vector of the target neuron when preprocessing is 'multiply_transpose'.
    """
    def __init__(self, dataset_path, transpose_weights=False, preprocessing=None, use_neurons=None,
                use_target_similarity_only: bool = False):
        """
        Initialize the dataset.
        
        Args:
            dataset_path (str): Path to directory containing saved model state_dict files.
                                Assumes files are named like 'seed-0', 'seed-1', etc. or similar pattern
                                loadable by torch.load.
            transpose_weights (bool): Whether to transpose the weight matrix. Defaults to False.
            preprocessing (str, optional): Preprocessing method to apply ('multiply_transpose' or 'dim_reduction')
            use_neurons (list, optional): List of specific neuron indices to use. If provided, only these
                                         neurons will be included in the dataset.
            use_target_similarity_only (bool, optional): If True and preprocessing is 'multiply_transpose', 
                                                     only the target neuron's similarity vector is returned. Defaults to False.
        """
        self.dataset_path = dataset_path
        self.layer = 'vit.head.weight' # Specific layer for ViT output head
        self.transpose_weights = transpose_weights
        self.preprocessing = preprocessing
        self.use_neurons = use_neurons
        self.use_target_similarity_only = use_target_similarity_only

        # Determine the list of model files
        self.model_files = sorted([f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))])
        if not self.model_files:
            raise FileNotFoundError(f"No model files found in {dataset_path}")

        # Load the first model to determine dimensions
        first_model_path = os.path.join(self.dataset_path, self.model_files[0])
        state_dict = torch.load(first_model_path, map_location='cpu') # Load to CPU to avoid potential CUDA issues
        
        if self.layer not in state_dict:
             # Try adding 'model.' prefix often used by Lightning or Timm
             potential_layer = 'model.' + self.layer
             if potential_layer in state_dict:
                 self.layer = potential_layer
             else:
                 raise KeyError(f"Layer '{self.layer}' (or 'model.{self.layer}') not found in state dict keys: {list(state_dict.keys())}")

        weights_shape = state_dict[self.layer].shape

        if not transpose_weights:
            # Output layer shape: [num_classes, embedding_dim]
            self.num_classes = weights_shape[0]
        else:
            # Shape after transpose: [embedding_dim, num_classes]
            self.num_classes = weights_shape[1]
            
        # If we're using specific neurons, update the effective number of classes
        if self.use_neurons is not None:
            if not all(idx < self.num_classes for idx in self.use_neurons):
                 raise ValueError(f"One or more indices in use_neurons {self.use_neurons} are out of bounds for num_classes {self.num_classes}")
            self.effective_num_classes = len(self.use_neurons)
        else:
            self.effective_num_classes = self.num_classes

    def __len__(self):
        """
        Return the length of the dataset.
        
        Returns:
            int: Number of samples in the dataset (number of models × effective number of classes)
        """
        num_models = len(self.model_files)
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
        model_file_idx = idx // self.effective_num_classes
        neuron_idx = idx % self.effective_num_classes
        
        # If we're using specific neurons, map the neuron_idx to the actual neuron index
        if self.use_neurons is not None:
            class_idx = self.use_neurons[neuron_idx]
        else:
            class_idx = neuron_idx

        # load relevant data
        model_path = os.path.join(self.dataset_path, self.model_files[model_file_idx])
        state_dict = torch.load(model_path, map_location='cpu') # Load to CPU
        weights = state_dict[self.layer] 
        
        if self.transpose_weights:
            weights = weights.T
            
        # Filter to only use specified neurons if requested
        if self.use_neurons is not None:
            weights = weights[self.use_neurons, :]
            # Now class_idx needs to be mapped to its position in the filtered weights
            try:
                class_idx_in_filtered = self.use_neurons.index(class_idx)
            except ValueError:
                # This should not happen if validation in __init__ is correct, but as a safeguard:
                 raise ValueError(f"Target class index {class_idx} not found in use_neurons list: {self.use_neurons}")
        else:
            class_idx_in_filtered = class_idx

        # shuffle rows of weight matrix to hide the target class index
        # Swap the target class row with the first row
        tmp = weights[class_idx_in_filtered].clone()
        weights[class_idx_in_filtered] = weights[0]
        weights[0] = tmp

        # Shuffle the rest of the rows (excluding the first one, which is now the target)
        if weights.shape[0] > 1:  # Only shuffle if there's more than one row
            shuffle_indices = torch.randperm(weights.shape[0] - 1) + 1 # Indices from 1 to N-1
            # Apply permutation to rows starting from the second row
            original_order = torch.arange(1, weights.shape[0])
            permuted_rows = weights[original_order[shuffle_indices]]
            weights[1:] = permuted_rows


        # apply preprocessing 
        if self.preprocessing == 'multiply_transpose':
            # Normalize weights first
            weights_norm = weights / (torch.norm(weights, dim=1, keepdim=True) + 1e-8) # Add epsilon for stability
            # Then compute cosine similarities
            sim_matrix = weights_norm @ weights_norm.T
            if self.use_target_similarity_only:
                weights = sim_matrix[0:1, :] # Return only the first row (target neuron similarity) as a 2D tensor
            else:
                weights = sim_matrix # Return the full similarity matrix

            # permute weights columns (principal components)
            if weights.shape[1] > 1: # Only permute if there's more than one column
                weights = weights[:, torch.randperm(weights.shape[1])]

        # The target label is always 0 because we swapped the target neuron's weights to the first row
        target_label = torch.tensor([0], dtype=torch.long) 

        return weights.float(), target_label # Ensure output is float and label is long


class ViTOutputLayerDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the ViTOutputLayerDataset.
    
    Handles dataset creation, splitting, and DataLoaders for ViT output layer weights.
    
    Attributes:
        dataset_path: Path to directory containing saved model state_dict files.
        batch_size: Batch size for training/validation.
        num_workers: Number of workers for data loading.
        transpose_weights: Whether to transpose the weight matrix.
        preprocessing: Optional preprocessing method.
        use_neurons: Optional list of specific neuron indices to use.
        use_target_similarity_only: Whether to use only the target neuron's similarity vector.
        train_val_split_ratio: Ratio for splitting data into training and validation sets.
        input_dim: Input dimension (embedding size of the ViT). Automatically determined.
        output_dim: Output dimension (number of classes/neurons). Automatically determined.
    """
    def __init__(self, dataset_path, batch_size, num_workers, 
                 transpose_weights=False, preprocessing=None, use_neurons=None, 
                 use_target_similarity_only: bool = False,
                 train_val_split_ratio: float = 0.8):
        """
        Initialize the data module.
        
        Args:
            dataset_path (str): Path to directory containing saved model state_dict files.
            batch_size (int): Batch size for DataLoaders.
            num_workers (int): Number of workers for DataLoaders.
            transpose_weights (bool): Whether to transpose the weight matrix.
            preprocessing (str, optional): Preprocessing method to apply.
            use_neurons (list, optional): List of specific neuron indices to use.
            use_target_similarity_only (bool, optional): Passed to ViTOutputLayerDataset.
            train_val_split_ratio (float): Ratio for splitting data into training and validation sets (default: 0.8).
        """
        super().__init__()
        self.save_hyperparameters() # Saves init args to self.hparams

    def prepare_data(self):
        # Check if dataset path exists, etc. Can be used for downloads if needed.
        if not os.path.isdir(self.hparams.dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {self.hparams.dataset_path}")
        pass # No download needed for local files

    def setup(self, stage=None):
        """
        Set up the dataset and create train/validation splits.
        
        Creates a ViTOutputLayerDataset instance, determines input/output dimensions, 
        and splits it into train and validation subsets based on the split ratio.
        
        Args:
            stage (str, optional): Stage of setup ('fit', 'validate', 'test', 'predict')
        """
        if stage == 'fit' or stage is None:
            full_dataset = ViTOutputLayerDataset(
                dataset_path=self.hparams.dataset_path, 
                transpose_weights=self.hparams.transpose_weights, 
                preprocessing=self.hparams.preprocessing,
                use_neurons=self.hparams.use_neurons,
                use_target_similarity_only=self.hparams.use_target_similarity_only
            )

            # Determine input and output dimensions from the first sample
            sample_weights, _ = full_dataset[0]
            self.output_dim = full_dataset.effective_num_classes # Number of rows after filtering/shuffling
            
            if self.hparams.preprocessing == 'multiply_transpose':
                 if self.hparams.use_target_similarity_only:
                      self.input_dim = full_dataset.effective_num_classes # Input is similarity vector of size num_classes
                 else:
                      # Input is the full similarity matrix, flatten it or handle appropriately in model
                      # Let's assume the model expects the matrix shape [num_classes, num_classes]
                      self.input_dim = (full_dataset.effective_num_classes, full_dataset.effective_num_classes)
            else: # No preprocessing or unknown
                 self.input_dim = sample_weights.shape[1] # Embedding dimension is the input feature size


            # Split data
            total_len = len(full_dataset)
            train_size = int(total_len * self.hparams.train_val_split_ratio)
            val_size = total_len - train_size

            # Use deterministic splitting for reproducibility if needed
            generator = torch.Generator().manual_seed(42) # Use a fixed seed
            self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

            print(f"Dataset setup complete:")
            print(f"  Total samples: {total_len}")
            print(f"  Train samples: {len(self.train_dataset)}")
            print(f"  Validation samples: {len(self.val_dataset)}")
            print(f"  Determined input dim: {self.input_dim}")
            print(f"  Effective output dim (classes): {self.output_dim}")


        # Assign test dataset if available (using validation set for now as placeholder)
        if stage == 'test' or stage is None:
             if not hasattr(self, 'val_dataset'): # If setup wasn't called for 'fit'
                  self.setup('fit') # Run fit setup first
             self.test_dataset = self.val_dataset # Reusing validation set for testing

        # Assign predict dataset if available (using validation set for now as placeholder)
        if stage == 'predict' or stage is None:
             if not hasattr(self, 'val_dataset'): # If setup wasn't called for 'fit'
                  self.setup('fit') # Run fit setup first
             self.predict_dataset = self.val_dataset # Reusing validation set for prediction


    def train_dataloader(self):
        if not hasattr(self, 'train_dataset'):
             raise RuntimeError("Train dataset not initialized. Call setup('fit') first.")
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True, # Improves data transfer speed to GPU
            drop_last=True, # Drop last incomplete batch during training
            persistent_workers=True if self.hparams.num_workers > 0 else False # Keep workers alive
        )

    def val_dataloader(self):
        if not hasattr(self, 'val_dataset'):
             raise RuntimeError("Validation dataset not initialized. Call setup('fit') or setup('validate') first.")
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=False, # Don't drop last batch for validation/testing
            persistent_workers=True if self.hparams.num_workers > 0 else False
        )

    def test_dataloader(self):
        if not hasattr(self, 'test_dataset'):
             # Attempt to set it up if not already done
             self.setup('test')
             if not hasattr(self, 'test_dataset'):
                 raise RuntimeError("Test dataset not initialized. Call setup('test') first.")

        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True if self.hparams.num_workers > 0 else False
        )
        
    def predict_dataloader(self):
        if not hasattr(self, 'predict_dataset'):
             # Attempt to set it up if not already done
             self.setup('predict')
             if not hasattr(self, 'predict_dataset'):
                 raise RuntimeError("Predict dataset not initialized. Call setup('predict') first.")
                 
        return DataLoader(
            dataset=self.predict_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True if self.hparams.num_workers > 0 else False
        ) 