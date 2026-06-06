"""Neuron ablation experiment for gram matrix decoder.

This experiment systematically reduces the number of output neurons (from 10 down to 2)
to investigate how relational structure complexity affects decoding accuracy using the
gram matrix decoder approach. This replicates the ablation study originally done with
the self-attention decoder.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from config import (
    REFERENCE_SEEDS, TOLERANCE, RANDOM_SEED, BASEDIR,
    REFERENCE_MODEL_TYPE, REFERENCE_DATASET_TYPE, REFERENCE_HIDDEN_DIM, REFERENCE_UNTRAINED,
    EVAL_MODEL_TYPE, EVAL_DATASET_TYPE, EVAL_HIDDEN_DIM, EVAL_UNTRAINED
)
from data_loader import load_last_layer, gram, frob
from permutation import permutation_iterator, get_permutation_count
import math

# Use dropout networks as the base, matching the main geometric-matching result
# (Figure 3 / §3.1.2), which reports 100% for dropout. The ablation is the same
# experiment restricted to k of the 10 output neurons.
DROPOUT_MODEL_FMT = "fully_connected_dropout-mnist-hidden_dim_[50,50]/seed-{seed}"


def position_accuracy_for_subset(G_ref, W_test_subset, k, tolerance):
    """Hard position-wise accuracy of the argmin permutation for a k-neuron subset.

    Mirrors experiment.evaluate_seed + analysis.calculate_permutation_accuracy:
    find the permutation(s) minimizing Frobenius distance to the reference, then
    score the fraction of neuron positions they place correctly (averaging over
    ties). This is the SAME metric as the main result in §3.1.2 / Figure 3.
    """
    identity = np.arange(k)

    # The iterator yields every permutation (including identity), so let it
    # populate the argmin set rather than pre-seeding identity (which would
    # double-count it and inflate tie cases such as k=2).
    best_dist = np.inf
    best_perms = []
    for perm in permutation_iterator(k):
        p = np.asarray(perm, dtype=int)
        d_perm = frob(G_ref, gram(W_test_subset[p]))
        if d_perm < best_dist - tolerance:
            best_dist = d_perm
            best_perms = [p]
        elif abs(d_perm - best_dist) <= tolerance:
            best_perms.append(p)

    return float(np.mean([(np.asarray(p) == identity).mean() for p in best_perms]))


def run_neuron_ablation_experiment(
    neuron_counts=None,
    test_seeds=range(10, 15),
    reference_seeds=range(0, 5),
    save_results=True
):
    """
    Run gram matrix decoder ablation study with different neuron counts using ALL permutations.
    
    Args:
        neuron_counts (list): List of neuron counts to test (e.g., [2, 3, 4, 5, 6, 7, 8, 9, 10])
        test_seeds (range): Seeds to use for testing
        reference_seeds (range): Seeds to use for building reference geometry
        save_results (bool): Whether to save results to CSV
    
    Returns:
        pandas.DataFrame: Results with columns [neuron_count, seed, validation_accuracy, relative_performance]
    """
    if neuron_counts is None:
        neuron_counts = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    print("=" * 70)
    print("GRAM MATRIX DECODER - NEURON ABLATION EXPERIMENT")
    print("=" * 70)
    print(f"Testing neuron counts: {neuron_counts}")
    print(f"Test seeds: {list(test_seeds)}")
    print(f"Reference seeds: {list(reference_seeds)}")
    print("Using ALL permutations for each neuron count")
    print()
    
    results = []
    
    for neuron_count in neuron_counts:
        print(f"\n{'='*50}")
        print(f"TESTING WITH {neuron_count} NEURONS")
        print(f"{'='*50}")
        
        # Build reference geometry with reduced neurons (dropout networks)
        print("Building reference geometry...")
        ref_grams = []
        for seed in reference_seeds:
            W_ref = load_last_layer(seed, model_fmt=DROPOUT_MODEL_FMT)
            # Take only first neuron_count neurons
            W_ref_subset = W_ref[:neuron_count]
            ref_grams.append(gram(W_ref_subset))
        
        G_ref = np.mean(ref_grams, axis=0)
        print(f"  → Reference geometry built with {neuron_count} neurons")
        
        # Calculate random guessing baseline and show permutation count
        random_baseline = 1.0 / neuron_count
        total_perms = get_permutation_count(neuron_count)
        print(f"  → Random guessing baseline: {random_baseline:.4f}")
        print(f"  → Testing all {total_perms:,} permutations ({neuron_count}!)")
        
        # Evaluate each test seed
        seed_results = []
        for seed in test_seeds:
            W_test = load_last_layer(seed, model_fmt=DROPOUT_MODEL_FMT)
            W_test_subset = W_test[:neuron_count]

            # Hard position-wise accuracy of the argmin permutation (same metric
            # as §3.1.2 / Figure 3). Chance level is 1/k, so relative = acc * k.
            validation_accuracy = position_accuracy_for_subset(
                G_ref, W_test_subset, neuron_count, TOLERANCE
            )
            relative_performance = validation_accuracy / random_baseline

            seed_results.append({
                'neuron_count': neuron_count,
                'seed': seed,
                'validation_accuracy': validation_accuracy,
                'relative_performance': relative_performance,
            })

            print(f"  Seed {seed:2d}: acc={validation_accuracy:.4f} "
                  f"rel={relative_performance:.2f}x")
        
        results.extend(seed_results)
        
        # Calculate summary statistics for this neuron count
        accuracies = [r['validation_accuracy'] for r in seed_results]
        relatives = [r['relative_performance'] for r in seed_results]
        
        print(f"\n  SUMMARY for {neuron_count} neurons:")
        print(f"    Avg accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"    Avg relative: {np.mean(relatives):.2f}x ± {np.std(relatives):.2f}x")
        
        # Special check for 2-neuron case
        if neuron_count == 2:
            avg_rel = np.mean(relatives)
            if abs(avg_rel - 1.0) < 0.1:  # Within 10% of 1.0x
                print(f"    ✓ 2-neuron case performs near random chance ({avg_rel:.2f}x), confirming hypothesis")
            else:
                print(f"    ⚠ 2-neuron case performance ({avg_rel:.2f}x) differs from expected 1.0x")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    if save_results:
        output_file = Path(__file__).parent / "runs" / "gram_neuron_ablation_results.csv"
        output_file.parent.mkdir(exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"\n✓ Results saved to: {output_file}")
    
    return df


def print_summary_comparison(df):
    """Print a summary comparing results across neuron counts."""
    print("\n" + "="*70)
    print("SUMMARY COMPARISON ACROSS NEURON COUNTS")
    print("="*70)
    
    summary = df.groupby('neuron_count').agg({
        'validation_accuracy': ['mean', 'std'],
        'relative_performance': ['mean', 'std']
    }).round(4)
    
    print("Neuron Count | Validation Accuracy    | Relative Performance")
    print("             | Mean ± Std             | Mean ± Std")
    print("-" * 60)
    
    for neuron_count in sorted(df['neuron_count'].unique()):
        acc_mean = summary.loc[neuron_count, ('validation_accuracy', 'mean')]
        acc_std = summary.loc[neuron_count, ('validation_accuracy', 'std')]
        rel_mean = summary.loc[neuron_count, ('relative_performance', 'mean')]
        rel_std = summary.loc[neuron_count, ('relative_performance', 'std')]
        
        print(f"     {neuron_count:2d}      | {acc_mean:.4f} ± {acc_std:.4f}      | {rel_mean:.2f}x ± {rel_std:.2f}x")
    
    print()
    
    # Highlight key findings
    two_neuron_rel = summary.loc[2, ('relative_performance', 'mean')]
    max_neuron = df['neuron_count'].max()
    max_neuron_rel = summary.loc[max_neuron, ('relative_performance', 'mean')]
    
    print("KEY FINDINGS:")
    print(f"• 2-neuron relative performance: {two_neuron_rel:.2f}x (should be ~1.0x)")
    print(f"• {max_neuron}-neuron relative performance: {max_neuron_rel:.2f}x")
    print(f"• Performance ratio (10-neuron/2-neuron): {max_neuron_rel/two_neuron_rel:.2f}x")


if __name__ == "__main__":
    # Run the experiment
    results_df = run_neuron_ablation_experiment()
    
    # Print summary comparison
    print_summary_comparison(results_df)