#!/usr/bin/env python3
"""
Run mixed dataset input pixel experiments.

This script runs experiments where the decoder is trained on models from one dataset
(MNIST) and evaluated on models from another dataset (Fashion-MNIST).
"""

import sys
import os

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from decoder.experiments import run_mixed_datasets_inputpixels

if __name__ == "__main__":
    print("Running mixed datasets input pixel experiments...")
    print("Training on MNIST models, validating on Fashion-MNIST models")
    
    # Run the experiment with default parameters
    run_mixed_datasets_inputpixels(
        num_seeds=3,  # Start with 3 seeds for testing
        project_name="inputpixels-mnist-to-fashionmnist",
        train_dataset_str='mnist',
        valid_dataset_str='fashionmnist',
        train_samples=800 * 784,  # 800 models × 784 pixels per model
        valid_samples=200 * 784,  # 200 models × 784 pixels per model  
        positional_encoding_type='dist_center'
    )
    
    print("Experiments completed!")