#!/usr/bin/env python3
"""
Run dataset classification experiments.

This script runs experiments to classify whether a neural network was trained 
on MNIST or Fashion-MNIST based on cosine similarities of output neurons.
"""

import sys
import os

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from decoder.experiments import run_dataset_classification_experiments

if __name__ == "__main__":
    print("Running dataset classification experiments...")
    print("Classifying MNIST vs Fashion-MNIST based on output neuron cosine similarities")
    
    # Run the experiment with default parameters
    run_dataset_classification_experiments(
        num_seeds=3,  # Start with 3 seeds for testing
        project_name="dataset-classification-cosine-similarities",
        train_samples=800,  # 400 MNIST + 400 Fashion-MNIST models
        valid_samples=200   # 100 MNIST + 100 Fashion-MNIST models
    )
    
    print("Experiments completed!")