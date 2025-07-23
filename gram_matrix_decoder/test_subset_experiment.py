#!/usr/bin/env python3
"""Test script for the new subset-based regression experiment."""

from input_layer_experiment import run_input_layer_experiment_subset

if __name__ == "__main__":
    # Test with k=10 and 100000 random subsets
    results = run_input_layer_experiment_subset(k=10, n_random_subsets=10000000)
    
    print(f"\nCompleted experiment with {len(results)} seeds")
    print("First result keys:", list(results[0].keys()))