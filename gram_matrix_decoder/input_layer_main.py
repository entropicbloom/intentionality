"""Main script to run input layer gram matrix decoder experiments."""

from input_layer_experiment import run_input_layer_experiment
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def plot_regression_results(results):
    """Plot regression results with R² scores."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Frobenius distances comparison
    ax1 = axes[0]
    identity_dist = [r['identity_frobenius_distance'] for r in results]
    first_random_dist = [r['first_random_frobenius_distance'] for r in results]
    best_dist = [r['best_frobenius_distance'] for r in results]
    seeds = [r['seed'] for r in results]
    
    ax1.scatter(seeds, identity_dist, label='Identity', alpha=0.7, s=50)
    ax1.scatter(seeds, first_random_dist, label='First Random', alpha=0.7, s=50)
    ax1.scatter(seeds, best_dist, label='Best Permutation', alpha=0.7, s=50)
    ax1.set_xlabel('Seed')
    ax1.set_ylabel('Frobenius Distance')
    ax1.set_title('Frobenius Distance Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: R² distribution across seeds
    ax2 = axes[1]
    r2_values = [r2_score(r['true_distances'], r['predicted_distances']) for r in results]
    ax2.hist(r2_values, bins=10, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('R² Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'R² Distribution (Mean = {np.mean(r2_values):.3f} ± {np.std(r2_values):.3f})')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run input layer experiments."""
    print("Running Input Layer Gram Matrix Decoder Experiment...\n")
    
    # Run the experiment
    results = run_input_layer_experiment()
    
    # Calculate R² scores for summary
    r2_scores = [r2_score(r['true_distances'], r['predicted_distances']) for r in results]
    print(f"Average R² Score: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    
    # Plot results
    plot_regression_results(results)
    
    return results


if __name__ == "__main__":
    main()