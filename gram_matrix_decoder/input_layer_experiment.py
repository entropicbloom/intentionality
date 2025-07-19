"""Input layer gram matrix decoder experiment for distance-from-center regression."""

import numpy as np
import time
from sklearn.metrics import r2_score
from config import REFERENCE_SEEDS, TEST_SEEDS, N_RANDOM_PERMS, RANDOM_SEED
from input_layer_data_loader import load_first_layer, gram_input, get_distance_labels, frob


def build_reference_geometry_input(reference_seeds=None, reference_model_fmt=None, basedir=None):
    """Build reference Gram matrix for input layer (mean over reference seeds)."""
    if reference_seeds is None:
        reference_seeds = REFERENCE_SEEDS
    
    print("Building reference geometry for input layer...")
    ref_grams = []
    for s in reference_seeds:
        W = load_first_layer(s, model_type="reference", model_fmt=reference_model_fmt, basedir=basedir)
        ref_grams.append(gram_input(W))
    
    G_ref = np.mean(ref_grams, axis=0)  # Average Gram matrix
    N = G_ref.shape[0]  # Number of input neurons (784 for MNIST)
    
    print(f"  → Averaged over {len(reference_seeds)} seeds, N = {N} input neurons")
    print(f"  → Will sample {N_RANDOM_PERMS:,} random permutations per test seed\n")
    
    return G_ref, N


def evaluate_seed_regression(seed: int, G_ref: np.ndarray, N: int, eval_model_fmt=None, basedir=None):
    """Evaluate regression task for a single test seed."""
    np.random.seed(RANDOM_SEED + seed)  # Reproducible random permutations
    
    # Load test model weights
    W_test = load_first_layer(seed, model_type="eval", model_fmt=eval_model_fmt, basedir=basedir)
    G_test = gram_input(W_test)
    
    # Get ground truth distance labels
    true_distances = get_distance_labels()
    
    # Identity permutation distance (for comparison only)
    identity_distance = frob(G_ref, G_test)
    
    # Initialize with first random permutation
    first_random_perm = np.random.permutation(N)
    G_perm = G_test[np.ix_(first_random_perm, first_random_perm)]
    first_random_distance = frob(G_ref, G_perm)
    
    # Start search with first random permutation
    best_permutation = first_random_perm.copy()
    best_distance = first_random_distance
    
    # Search remaining random permutations
    start_time = time.time()
    for i in range(N_RANDOM_PERMS - 1):
        # Random permutation
        perm = np.random.permutation(N)
        
        # Apply permutation to test Gram matrix more efficiently
        # Instead of G_test[perm][:, perm], use advanced indexing
        G_perm = G_test[np.ix_(perm, perm)]
        
        # Calculate Frobenius distance
        perm_distance = frob(G_ref, G_perm)
        
        # Update best if this permutation is better
        if perm_distance < best_distance:
            best_distance = perm_distance
            best_permutation = perm
        
        # Progress tracking (print every 1000 permutations)
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (N_RANDOM_PERMS - 1 - i) / rate
            print(f"    Seed {seed}: {i + 1}/{N_RANDOM_PERMS - 1} permutations "
                  f"({rate:.0f} perm/sec, {remaining:.0f}s remaining)")
    
    total_time = time.time() - start_time
    
    # Get predicted distances using different permutations
    predicted_distances = true_distances[best_permutation]
    identity_predicted = true_distances  # Identity permutation
    first_random_predicted = true_distances[first_random_perm]
    
    # Calculate regression metrics
    mse = np.mean((true_distances - predicted_distances) ** 2)
    mae = np.mean(np.abs(true_distances - predicted_distances))
    r2 = r2_score(true_distances, predicted_distances)
    
    # Identity permutation metrics for comparison
    identity_mse = np.mean((true_distances - true_distances) ** 2)  # Should be 0
    identity_mae = np.mean(np.abs(true_distances - true_distances))  # Should be 0
    identity_r2 = r2_score(true_distances, true_distances)  # Should be 1
    
    improvement = identity_distance > best_distance
    
    print(f"Seed {seed:3d}: "
          f"Frob dist = {best_distance:.4f} (identity: {identity_distance:.4f})  |  "
          f"R² = {r2:.4f}, MSE = {mse:.4f}  |  "
          f"Time: {total_time:.1f}s ({N_RANDOM_PERMS/total_time:.0f} perm/sec)  "
          f"→ {'✓ improved' if improvement else '✗ no improvement'}")
    
    return {
        'seed': seed,
        'best_permutation': best_permutation,
        'best_frobenius_distance': best_distance,
        'first_random_frobenius_distance': first_random_distance,
        'identity_frobenius_distance': identity_distance,
        'predicted_distances': predicted_distances,
        'identity_predicted': identity_predicted,
        'first_random_predicted': first_random_predicted,
        'true_distances': true_distances,
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'improvement': improvement
    }


def run_input_layer_experiment():
    """Run the input layer gram matrix decoder experiment for distance regression."""
    print("=== Input Layer Gram Matrix Decoder (Distance-from-Center Regression) ===\n")
    
    # Build reference geometry
    G_ref, N = build_reference_geometry_input()
    
    # Evaluate each test seed
    results = []
    improved_count = 0
    
    for seed in TEST_SEEDS:
        result = evaluate_seed_regression(seed, G_ref, N)
        results.append(result)
        if result['improvement']:
            improved_count += 1
    
    # Summary statistics
    all_mse = [r['mse'] for r in results]
    all_mae = [r['mae'] for r in results]
    all_r2 = [r['r2'] for r in results]
    
    print(f"\n=== Summary ===")
    print(f"Seeds with improved Frobenius distance: {improved_count} / {len(TEST_SEEDS)}")
    print(f"Average R² Score: {np.mean(all_r2):.4f} ± {np.std(all_r2):.4f}")
    print(f"Average MSE: {np.mean(all_mse):.4f} ± {np.std(all_mse):.4f}")
    print(f"Average MAE: {np.mean(all_mae):.4f} ± {np.std(all_mae):.4f}")
    
    return results