"""Main script to run the gram matrix decoder experiment."""

from experiment import run_experiment
from analysis import calculate_permutation_accuracy, plot_distance_distribution
from comparison import run_comparison_experiment


def run_standard_experiment():
    """Run the standard gram matrix decoder experiment."""
    # Run the main experiment
    best_permutations, distances, C = run_experiment()
    
    # Calculate accuracy
    accuracy = calculate_permutation_accuracy(best_permutations, C)
    
    # Plot results (using first seed's distances)
    if distances and len(distances[0]) > 0:
        plot_distance_distribution(distances, seed_idx=0)
    
    return best_permutations, distances, C, accuracy


def main():
    """Main function to choose which experiment to run."""
    print("Choose experiment to run:")
    print("1. Standard experiment")
    print("2. Model comparison experiment")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == '1':
        return run_standard_experiment()
    elif choice == '2':
        return run_comparison_experiment()
    else:
        print("Invalid choice. Running standard experiment.")
        return run_standard_experiment()


if __name__ == "__main__":
    main()