"""Main script to run the gram matrix decoder experiment."""

from experiment import run_experiment
from analysis import calculate_permutation_accuracy, plot_distance_distribution


def main():
    """Run the complete gram matrix decoder experiment."""
    # Run the main experiment
    best_permutations, distances, C = run_experiment()
    
    # Calculate accuracy
    calculate_permutation_accuracy(best_permutations, C)
    
    # Plot results (using first seed's distances)
    if distances and len(distances[0]) > 0:
        plot_distance_distribution(distances, seed_idx=0)
    
    return best_permutations, distances, C


if __name__ == "__main__":
    main()