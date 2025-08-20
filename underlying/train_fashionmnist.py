#!/usr/bin/env python3
"""
Script to train Fashion-MNIST models with both vanilla backprop and dropout variants.
Uses the existing main.py infrastructure by temporarily modifying the CONFIG.
"""

import sys
import os
from main import CONFIG, run, get_dir_path, get_start_seed, load_config, MAX_SEEDS

# Base configuration for Fashion-MNIST
BASE_CONFIG = {
    'dataset_class_str': 'fashionmnist',
    'batch_size': 256,
    'num_epochs': 4,
    'learning_rate': 0.001,
    'num_workers': 4,
    'num_classes': 10,
    'hidden_dim': [50, 50],
    'varying_dim_bounds': None
}

# Model variants
MODEL_VARIANTS = {
    'vanilla': 'fully_connected',
    'dropout': 'fully_connected_dropout'
}

MODELS_DIR = 'saved_models/'

def train_variant(variant_name, max_seeds=10):
    """Train a specific variant (vanilla or dropout) on Fashion-MNIST."""
    print(f"\n=== Training {variant_name} variant on Fashion-MNIST ===")
    
    # Create config for this variant
    config = BASE_CONFIG.copy()
    config['model_class_str'] = MODEL_VARIANTS[variant_name]
    
    # Get the path and create directory if it doesn't exist
    path = get_dir_path(config['model_class_str'], config['dataset_class_str'], 
                       config['num_epochs'], config['hidden_dim'], config['varying_dim_bounds'], MODELS_DIR)
    if not os.path.exists(path):
        os.makedirs(path)

    # Check if config exists and matches
    existing_config = load_config(path)
    if existing_config is not None:
        if existing_config != config:
            print(f"Warning: Existing config in {path} does not match current config.")
            print(f"Existing: {existing_config}")
            print(f"Current: {config}")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return
    else:
        # Save config if it doesn't exist
        with open(path + 'train_config.txt', 'w') as f:
            print(config, file=f)

    start_seed = get_start_seed(path)
    print(f"Starting training from seed {start_seed}")

    for seed in range(start_seed, min(start_seed + max_seeds, MAX_SEEDS)):
        print(f"Training {variant_name} model with seed {seed}")
        run(**config, seed=seed)
        
    print(f"Completed training {max_seeds} seeds for {variant_name} variant")

if __name__ == '__main__':
    max_seeds = 1000
    
    if len(sys.argv) > 1:
        if sys.argv[1] in MODEL_VARIANTS:
            train_variant(sys.argv[1], max_seeds)
        else:
            print("Usage: python train_fashionmnist.py [vanilla|dropout]")
            print("If no argument provided, trains both variants")
    else:
        # Train both variants
        for variant in MODEL_VARIANTS:
            train_variant(variant, max_seeds)