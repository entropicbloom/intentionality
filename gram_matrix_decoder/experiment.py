"""Main experiment runner for gram matrix decoder."""

import numpy as np
from config import REFERENCE_SEEDS, TEST_SEEDS, SAVE_DISTANCES, TOLERANCE, ALL_PERMS
from data_loader import load_last_layer, gram, frob
from permutation import permutation_iterator, get_permutation_count


def build_reference_geometry(reference_seeds=None, reference_model_fmt=None, basedir=None, 
                           all_perms=None, n_random_perms=None):
    """Build reference Gram matrix (mean over reference seeds)."""
    # Use defaults from config if not provided
    if reference_seeds is None:
        reference_seeds = REFERENCE_SEEDS
    if all_perms is None:
        all_perms = ALL_PERMS
    
    print("Building reference geometry …")
    ref_grams = [gram(load_last_layer(s, model_type="reference", model_fmt=reference_model_fmt, basedir=basedir)) 
                 for s in reference_seeds]
    G_ref = np.mean(ref_grams, axis=0)       # still PSD, see discussion
    
    C = G_ref.shape[0]                       # number of classes / output neurons
    print(f"  → Averaged over {len(reference_seeds)} seeds, C = {C}\n")
    
    total_perms = get_permutation_count(C)
    if all_perms:
        print(f"Evaluating ALL permutations: {total_perms:,} possibilities per seed - may take a while!\n")
    else:
        print(f"Evaluating {total_perms:,} random permutations per seed.\n")
    
    return G_ref, C


def evaluate_seed(seed: int, G_ref: np.ndarray, C: int, eval_model_fmt=None, basedir=None,
                  save_distances=None, tolerance=None, all_perms=None):
    """Evaluate a single test seed against reference geometry."""
    # Use defaults from config if not provided
    if save_distances is None:
        save_distances = SAVE_DISTANCES
    if tolerance is None:
        tolerance = TOLERANCE
    if all_perms is None:
        all_perms = ALL_PERMS
    
    W_test = load_last_layer(seed, model_type="eval", model_fmt=eval_model_fmt, basedir=basedir)
    G_test = gram(W_test)
    
    d_true = frob(G_ref, G_test)
    
    # Track best distances and permutations
    best_dist = d_true
    best_perms = [np.arange(C)]        # start with the true ordering
    distances_current_seed = []
    
    worse_count = 0
    for p in permutation_iterator(C):
        p = np.asarray(p, dtype=int)
        W_perm = W_test[p]
        G_perm = gram(W_perm)
        d_perm = frob(G_ref, G_perm)
        
        if save_distances:
            distances_current_seed.append(d_perm)
        
        # Update running best
        if d_perm < best_dist - tolerance:
            best_dist = d_perm
            best_perms = [p]    
            worse_count += 1
        elif abs(d_perm - best_dist) <= tolerance:
            best_perms.append(p)
    
    is_best = (worse_count == 0)
    
    perm_count = get_permutation_count(C)
    perm_str = 'all' if all_perms else perm_count
    
    print(f"Seed {seed:3d}: "
          f"d_true = {d_true:8.4f}  |  "
          f"perms better = {worse_count:4d} / "
          f"{perm_str}  "
          f"→ {'✓ best' if is_best else '✗ not best'}")
    
    return is_best, best_perms, distances_current_seed


def run_experiment():
    """Run the full gram matrix decoder experiment."""
    G_ref, C = build_reference_geometry()
    
    hit_counter = 0
    best_permutations = []
    distances = []
    
    for seed in TEST_SEEDS:
        is_best, best_perms, distances_current_seed = evaluate_seed(seed, G_ref, C)
        
        hit_counter += is_best
        best_permutations.append(best_perms)
        distances.append(distances_current_seed)
    
    print(f"\nOriginal ordering was the unique best in "
          f"{hit_counter} / {len(TEST_SEEDS)} test seeds.")
    
    return best_permutations, distances, C