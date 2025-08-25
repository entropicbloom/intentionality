import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

def main():
    # Add the underlying directory to the path to import analysis_utils
    sys.path.insert(0, str(Path(__file__).parent / "../underlying"))

    # Change to underlying directory so the relative path works
    import os
    os.chdir(Path(__file__).parent / "../underlying")

    import analysis_utils

    # Set parameters
    hidden_dim = [50, 50]
    dataset_type = "mnist"
    num_seeds = 10

    # Dictionary to store results for each model (match gram matrix order)
    results = {
        "untrained": [],
        "no_dropout": [],  # renamed from fully_connected for consistency
        "dropout": []      # renamed from fully_connected_dropout for consistency
    }

    print("Evaluating models across seeds...")

    # Run evaluation for each model across seeds
    for seed in range(num_seeds):
        print(f"Seed {seed}...")
        
        # Untrained model
        accuracy = analysis_utils.test_model("fully_connected", dataset_type, seed, "-untrained", hidden_dim)
        results["untrained"].append(accuracy)
        
        # Fully connected model (no dropout)
        accuracy = analysis_utils.test_model("fully_connected", dataset_type, seed, "", hidden_dim)
        results["no_dropout"].append(accuracy)
        
        # Fully connected with dropout
        accuracy = analysis_utils.test_model("fully_connected_dropout", dataset_type, seed, "", hidden_dim)
        results["dropout"].append(accuracy)

    # Calculate means and standard deviations (convert from percentage to proportion)
    model_names = list(results.keys())
    means = [np.mean(results[model]) / 100.0 for model in model_names]  # Convert to 0-1 scale
    stds = [np.std(results[model]) / 100.0 for model in model_names]   # Convert to 0-1 scale

    # Create bar chart (match gram matrix style)
    plt.figure(figsize=(10, 6))

    # Blue-green color palette to match gram matrix plots
    colors = ["#2980b9", "#16a085", "#8e44ad"]  # Blue, Teal, Purple (same as gram matrix)

    # Create bar chart with error bars (match gram matrix style)
    bars = plt.bar(model_names, means, yerr=stds, 
                   color=colors, alpha=0.7, 
                   capsize=5, error_kw={'linewidth': 2})

    # Customize the plot (match gram matrix style exactly)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.title('Underlying Model Performance on MNIST', fontsize=14, pad=20)
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)

    # Add value labels on bars (match gram matrix style)
    for bar, mean_val, std_val in zip(bars, means, stds):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_val + 0.02,
                f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')

    # Adjust layout and save (match gram matrix style)
    plt.tight_layout()
    plt.savefig('underlying_mnist_performance_comparison_real.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary table (match gram matrix style)
    print("\n" + "="*50)
    print("UNDERLYING MNIST MODEL PERFORMANCE SUMMARY")
    print("="*50)
    print(f"{'Model':<15} {'Accuracy':<12} {'Std Dev':<12}")
    print("-"*50)
    for name, mean_val, std_val in zip(model_names, means, stds):
        print(f"{name:<15} {mean_val:.3f}        {std_val:.3f}")
    print("="*50)
    print(f"Results across {num_seeds} seeds")

if __name__ == '__main__':
    main()