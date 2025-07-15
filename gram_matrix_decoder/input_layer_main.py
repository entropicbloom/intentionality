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
    
    # Plot 2: Relative improvement in distance-to-center prediction
    ax2 = axes[1]
    
    # Calculate relative improvement for each seed
    relative_improvements = []
    
    for result in results:
        # Average distance between first random predictions and true distances
        first_random_error = np.mean(np.abs(result['first_random_predicted'] - result['true_distances']))
        
        # Average distance between best permutation predictions and true distances
        best_error = np.mean(np.abs(result['predicted_distances'] - result['true_distances']))
        
        # Relative improvement: (random_error - best_error) / random_error
        relative_improvement = (first_random_error - best_error) / first_random_error
        relative_improvements.append(relative_improvement)
    
    ax2.scatter(seeds, relative_improvements, alpha=0.7, s=50, c='green')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No improvement')
    ax2.set_xlabel('Seed')
    ax2.set_ylabel('Relative Improvement')
    ax2.set_title('Relative Improvement in Distance-to-Center Prediction\n(First Random → Best Permutation)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Format y-axis as percentage
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
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
    
    # Calculate relative improvements
    relative_improvements = []
    for result in results:
        first_random_error = np.mean(np.abs(result['first_random_predicted'] - result['true_distances']))
        best_error = np.mean(np.abs(result['predicted_distances'] - result['true_distances']))
        relative_improvement = (first_random_error - best_error) / first_random_error
        relative_improvements.append(relative_improvement)
    
    print(f"Average Relative Improvement: {np.mean(relative_improvements):.2%} ± {np.std(relative_improvements):.2%}")
    
    # Plot results
    plot_regression_results(results)
    
    return results


if __name__ == "__main__":
    main()