"""Model comparison experiments for gram matrix decoder."""

import matplotlib.pyplot as plt
import numpy as np
import config
from experiment import build_reference_geometry, evaluate_seed
from analysis import calculate_permutation_accuracy


def run_comparison_experiment():
    """Run comparison experiment for untrained, no-dropout, and dropout models."""
    import importlib
    import data_loader
    
    # Store original config values
    orig_ref_model_type = config.REFERENCE_MODEL_TYPE
    orig_eval_model_type = config.EVAL_MODEL_TYPE
    orig_ref_untrained = config.REFERENCE_UNTRAINED
    orig_eval_untrained = config.EVAL_UNTRAINED
    
    # Define model configurations in desired order
    model_configs = [
        {
            'name': 'untrained',
            'ref_model_type': 'fully_connected',
            'eval_model_type': 'fully_connected',
            'ref_untrained': True,
            'eval_untrained': True
        },
        {
            'name': 'no_dropout',
            'ref_model_type': 'fully_connected', 
            'eval_model_type': 'fully_connected',
            'ref_untrained': False,
            'eval_untrained': False
        },
        {
            'name': 'dropout',
            'ref_model_type': 'fully_connected_dropout',
            'eval_model_type': 'fully_connected_dropout', 
            'ref_untrained': False,
            'eval_untrained': False
        }
    ]
    
    accuracies = []
    accuracy_stds = []
    model_names = []
    
    print("Running comparison experiment across model types...\\n")
    
    try:
        for model_config in model_configs:
            print(f"Evaluating {model_config['name']} models:")
            print(f"  Config: {model_config}")
            
            # Update config for this model type
            config.REFERENCE_MODEL_TYPE = model_config['ref_model_type']
            config.EVAL_MODEL_TYPE = model_config['eval_model_type']
            config.REFERENCE_UNTRAINED = model_config['ref_untrained']
            config.EVAL_UNTRAINED = model_config['eval_untrained']
            
            # Update the format strings
            config.REFERENCE_UNTRAINED_SUFFIX = "-untrained" if config.REFERENCE_UNTRAINED else ""
            config.EVAL_UNTRAINED_SUFFIX = "-untrained" if config.EVAL_UNTRAINED else ""
            config.REFERENCE_MODEL_FMT = f"{config.REFERENCE_MODEL_TYPE}-{config.REFERENCE_DATASET_TYPE}{config.REFERENCE_UNTRAINED_SUFFIX}-hidden_dim_{config.REFERENCE_HIDDEN_DIM}/seed-{{seed}}"
            config.EVAL_MODEL_FMT = f"{config.EVAL_MODEL_TYPE}-{config.EVAL_DATASET_TYPE}{config.EVAL_UNTRAINED_SUFFIX}-hidden_dim_{config.EVAL_HIDDEN_DIM}/seed-{{seed}}"
            
            print(f"  Reference format: {config.REFERENCE_MODEL_FMT}")
            print(f"  Eval format: {config.EVAL_MODEL_FMT}")
            
            # Need to reload data_loader to pick up new format strings
            importlib.reload(data_loader)
            
            # Use existing functions
            G_ref, C = build_reference_geometry()
            
            hit_counter = 0
            best_permutations = []
            hit_rates_per_seed = []
            
            # Use subset of test seeds for faster comparison
            test_seeds_subset = list(config.TEST_SEEDS)[:5]
            
            for seed in test_seeds_subset:
                is_best, best_perms, _ = evaluate_seed(seed, G_ref, C)
                hit_counter += is_best
                best_permutations.append(best_perms)
                hit_rates_per_seed.append(float(is_best))
                print(f"    Seed {seed}: is_best={is_best}, num_best_perms={len(best_perms)}")
            
            hit_rate = hit_counter / len(test_seeds_subset)
            hit_rate_std = np.std(hit_rates_per_seed)
            print(f"  Hit rate: {hit_rate:.3f} ({hit_counter}/{len(test_seeds_subset)})")
            
            position_accuracy, position_accuracy_std = calculate_permutation_accuracy(best_permutations, C)
            
            accuracies.append(position_accuracy)
            accuracy_stds.append(position_accuracy_std)
            model_names.append(model_config['name'])
            print(f"  Position accuracy: {position_accuracy:.3f} ± {position_accuracy_std:.3f}\\n")
    
    finally:
        # Restore original config values
        config.REFERENCE_MODEL_TYPE = orig_ref_model_type
        config.EVAL_MODEL_TYPE = orig_eval_model_type
        config.REFERENCE_UNTRAINED = orig_ref_untrained
        config.EVAL_UNTRAINED = orig_eval_untrained
        config.REFERENCE_UNTRAINED_SUFFIX = "-untrained" if orig_ref_untrained else ""
        config.EVAL_UNTRAINED_SUFFIX = "-untrained" if orig_eval_untrained else ""
        config.REFERENCE_MODEL_FMT = f"{orig_ref_model_type}-{config.REFERENCE_DATASET_TYPE}{config.REFERENCE_UNTRAINED_SUFFIX}-hidden_dim_{config.REFERENCE_HIDDEN_DIM}/seed-{{seed}}"
        config.EVAL_MODEL_FMT = f"{orig_eval_model_type}-{config.EVAL_DATASET_TYPE}{config.EVAL_UNTRAINED_SUFFIX}-hidden_dim_{config.EVAL_HIDDEN_DIM}/seed-{{seed}}"
    
    # Create bar chart with error bars
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies, yerr=accuracy_stds, 
                   color=['orange', 'blue', 'green'], alpha=0.7, 
                   capsize=5, error_kw={'linewidth': 2})
    plt.ylabel('Position Accuracy')
    plt.title('Gram Matrix Decoder Position Accuracy Comparison')
    plt.ylim(0, 1.1)
    
    # Add value labels on bars
    for bar, acc, std in zip(bars, accuracies, accuracy_stds):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracies, accuracy_stds