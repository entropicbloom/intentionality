#!/usr/bin/env python3
"""Script to execute cross-architecture experiments for gram matrix decoder."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cross_architecture_experiment import run_all_cross_architecture_experiments, print_cross_architecture_summary, plot_cross_architecture_heatmap


def main():
    """Run comprehensive cross-architecture experiments with MNIST dropout models."""
    
    print("Cross-Architecture Gram Matrix Decoder Experiments")
    print("=" * 65)
    print("Testing all combinations of reference vs evaluation architectures")
    print("Available architectures: [50,50], [25,25], [100]")
    print("Model type: fully_connected_dropout")
    print("Dataset: MNIST")
    print("=" * 65)
    
    # Run all cross-architecture experiments
    all_results = run_all_cross_architecture_experiments(
        model_type='fully_connected_dropout',
        dataset_type='mnist'
    )
    
    # Print comprehensive summary
    print_cross_architecture_summary(all_results)
    
    # Generate heatmap visualization
    try:
        print("\nGenerating heatmap visualization...")
        accuracy_matrix, hit_rate_matrix = plot_cross_architecture_heatmap(all_results)
        print("Heatmaps saved as:")
        print("  - cross_architecture_heatmap.png (combined)")
        print("  - cross_architecture_heatmap_accuracy.png (accuracy only)")
        print("  - cross_architecture_heatmap_hit_rate.png (hit rate only)")
    except Exception as e:
        print(f"Warning: Could not generate heatmap - {e}")
    
    print("\nExperiment completed!")
    return all_results


if __name__ == "__main__":
    main()