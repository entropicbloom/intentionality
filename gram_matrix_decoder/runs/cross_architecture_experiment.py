"""Cross-architecture variation experiments for gram matrix decoder."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from experiment import build_reference_geometry, evaluate_seed
from analysis import calculate_permutation_accuracy


def run_cross_architecture_experiment(reference_architecture, eval_architectures, 
                                    model_type='fully_connected_dropout', 
                                    dataset_type='mnist', untrained=False):
    """
    Run gram matrix decoder experiments with cross-architecture decoding.
    
    Args:
        reference_architecture: Dict with 'name' and 'hidden_dim' for reference geometry
        eval_architectures: List of dicts with 'name' and 'hidden_dim' for evaluation
        model_type: Model type string
        dataset_type: Dataset type string  
        untrained: Whether to use untrained models
    
    Returns:
        Dict with results for each evaluation architecture
    """
    accuracies = []
    accuracy_stds = []
    hit_rates = []
    arch_names = []
    
    # Build reference geometry from reference architecture
    ref_hidden_dim_str = str(reference_architecture['hidden_dim']).replace(' ', '')
    untrained_suffix = "-untrained" if untrained else ""
    ref_model_fmt = f"{model_type}-{dataset_type}{untrained_suffix}-hidden_dim_{ref_hidden_dim_str}/seed-{{seed}}"
    
    print(f"Building reference geometry from: {reference_architecture['name']} {reference_architecture['hidden_dim']}")
    print(f"Reference model format: {ref_model_fmt}")
    
    try:
        G_ref, C = build_reference_geometry(
            reference_seeds=config.REFERENCE_SEEDS,
            reference_model_fmt=ref_model_fmt,
            basedir=config.BASEDIR,
            all_perms=config.ALL_PERMS
        )
    except Exception as e:
        print(f"ERROR building reference geometry: {e}")
        return {'error': f'Reference architecture failed: {e}'}
    
    print(f"Testing {len(eval_architectures)} evaluation architectures against reference")
    print(f"Model: {model_type}, Dataset: {dataset_type}, Untrained: {untrained}\n")
    
    for arch_config in eval_architectures:
        arch_name = arch_config['name']
        hidden_dim_list = arch_config['hidden_dim']
        hidden_dim_str = str(hidden_dim_list).replace(' ', '')
        
        print(f"Evaluating architecture: {arch_name} {hidden_dim_list}")
        
        # Build eval model format for this architecture
        eval_model_fmt = f"{model_type}-{dataset_type}{untrained_suffix}-hidden_dim_{hidden_dim_str}/seed-{{seed}}"
        
        print(f"  Eval model format: {eval_model_fmt}")
        
        try:
            
            hit_counter = 0
            best_permutations = []
            
            # Use subset of test seeds for faster comparison
            test_seeds_subset = list(config.TEST_SEEDS)[:5]
            
            for seed in test_seeds_subset:
                is_best, best_perms, _ = evaluate_seed(
                    seed, G_ref, C,
                    eval_model_fmt=eval_model_fmt,
                    basedir=config.BASEDIR,
                    save_distances=config.SAVE_DISTANCES,
                    tolerance=config.TOLERANCE,
                    all_perms=config.ALL_PERMS
                )
                hit_counter += is_best
                best_permutations.append(best_perms)
                print(f"    Seed {seed}: is_best={is_best}, num_best_perms={len(best_perms)}")
            
            hit_rate = hit_counter / len(test_seeds_subset)
            print(f"  Hit rate: {hit_rate:.3f} ({hit_counter}/{len(test_seeds_subset)})")
            
            position_accuracy, position_accuracy_std = calculate_permutation_accuracy(best_permutations, C)
            
            accuracies.append(position_accuracy)
            accuracy_stds.append(position_accuracy_std)
            hit_rates.append(hit_rate)
            arch_names.append(arch_name)
            print(f"  Position accuracy: {position_accuracy:.3f} ± {position_accuracy_std:.3f}\n")
            
        except Exception as e:
            print(f"  ERROR: {e}\n")
            # Add placeholder values for failed architectures
            accuracies.append(0.0)
            accuracy_stds.append(0.0)
            hit_rates.append(0.0)
            arch_names.append(f"{arch_name}_ERROR")
    
    return {
        'accuracies': accuracies,
        'accuracy_stds': accuracy_stds,
        'hit_rates': hit_rates,
        'arch_names': arch_names
    }


def run_all_cross_architecture_experiments(model_type='fully_connected_dropout', 
                                          dataset_type='mnist', **kwargs):
    """Run all cross-architecture experiments with available MNIST dropout architectures."""
    
    # Available architectures
    architectures = [
        {'name': '2L_50W', 'hidden_dim': [50, 50]},
        {'name': '2L_25W', 'hidden_dim': [25, 25]},
        {'name': '1L_100W', 'hidden_dim': [100]}
    ]
    
    all_results = {}
    
    for ref_arch in architectures:
        print(f"\n{'='*80}")
        print(f"REFERENCE ARCHITECTURE: {ref_arch['name']} {ref_arch['hidden_dim']}")
        print(f"{'='*80}")
        
        results = run_cross_architecture_experiment(
            reference_architecture=ref_arch,
            eval_architectures=architectures,  # Test all architectures against this reference
            model_type=model_type,
            dataset_type=dataset_type,
            **kwargs
        )
        
        all_results[ref_arch['name']] = results
    
    return all_results


def print_cross_architecture_summary(all_results):
    """Print a comprehensive summary of all cross-architecture results."""
    print("\n" + "="*100)
    print("COMPREHENSIVE CROSS-ARCHITECTURE RESULTS SUMMARY")
    print("="*100)
    print(f"{'Ref Arch':<12} {'Eval Arch':<12} {'Accuracy':<12} {'Std Dev':<12} {'Hit Rate':<12} {'Baseline?'}")
    print("-"*100)
    
    for ref_name, results in all_results.items():
        if 'error' in results:
            print(f"{ref_name:<12} {'ERROR':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A'}")
            continue
            
        for eval_name, acc, std, hit_rate in zip(results['arch_names'], results['accuracies'], 
                                               results['accuracy_stds'], results['hit_rates']):
            is_baseline = "YES" if ref_name == eval_name else "NO"
            print(f"{ref_name:<12} {eval_name:<12} {acc:.3f}        {std:.3f}        {hit_rate:.3f}        {is_baseline}")
    
    print("="*100)


def plot_cross_architecture_heatmap(all_results, save_name="cross_architecture_heatmap.png"):
    """Plot cross-architecture results as a heatmap."""
    
    # Get architecture names
    arch_names = ['2L_50W', '2L_25W', '1L_100W']
    
    # Initialize matrices for accuracy and hit rates
    accuracy_matrix = np.zeros((len(arch_names), len(arch_names)))
    hit_rate_matrix = np.zeros((len(arch_names), len(arch_names)))
    
    # Fill matrices
    for i, ref_name in enumerate(arch_names):
        if ref_name in all_results and 'error' not in all_results[ref_name]:
            results = all_results[ref_name]
            for j, eval_name in enumerate(arch_names):
                if eval_name in results['arch_names']:
                    idx = results['arch_names'].index(eval_name)
                    accuracy_matrix[i, j] = results['accuracies'][idx]
                    hit_rate_matrix[i, j] = results['hit_rates'][idx]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy heatmap
    sns.heatmap(accuracy_matrix, 
                xticklabels=arch_names, 
                yticklabels=arch_names,
                annot=True, 
                fmt='.3f', 
                cmap='viridis',
                vmin=0, vmax=1,
                ax=ax1)
    ax1.set_title('Position Accuracy\n(Reference → Evaluation)')
    ax1.set_xlabel('Evaluation Architecture')
    ax1.set_ylabel('Reference Architecture')
    
    # Hit rate heatmap
    sns.heatmap(hit_rate_matrix, 
                xticklabels=arch_names, 
                yticklabels=arch_names,
                annot=True, 
                fmt='.3f', 
                cmap='plasma',
                vmin=0, vmax=1,
                ax=ax2)
    ax2.set_title('Hit Rate\n(Reference → Evaluation)')
    ax2.set_xlabel('Evaluation Architecture')
    ax2.set_ylabel('Reference Architecture')
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save individual heatmaps as separate files
    # Accuracy heatmap only
    plt.figure(figsize=(8, 6))
    sns.heatmap(accuracy_matrix, 
                xticklabels=arch_names, 
                yticklabels=arch_names,
                annot=True, 
                fmt='.3f', 
                cmap='viridis',
                vmin=0, vmax=1)
    plt.title('Position Accuracy\n(Reference → Evaluation)')
    plt.xlabel('Evaluation Architecture')
    plt.ylabel('Reference Architecture')
    plt.tight_layout()
    accuracy_save_name = save_name.replace('.png', '_accuracy.png')
    plt.savefig(accuracy_save_name, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Hit rate heatmap only
    plt.figure(figsize=(8, 6))
    sns.heatmap(hit_rate_matrix, 
                xticklabels=arch_names, 
                yticklabels=arch_names,
                annot=True, 
                fmt='.3f', 
                cmap='plasma',
                vmin=0, vmax=1)
    plt.title('Hit Rate\n(Reference → Evaluation)')
    plt.xlabel('Evaluation Architecture')
    plt.ylabel('Reference Architecture')
    plt.tight_layout()
    hit_rate_save_name = save_name.replace('.png', '_hit_rate.png')
    plt.savefig(hit_rate_save_name, dpi=300, bbox_inches='tight')
    plt.close()
    
    return accuracy_matrix, hit_rate_matrix


def plot_results(results, title="Cross-Architecture Experiment Results", save_name="cross_architecture_results.png"):
    """Plot results with bar chart."""
    arch_names = results['arch_names']
    accuracies = results['accuracies']
    accuracy_stds = results['accuracy_stds']
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(arch_names, accuracies, yerr=accuracy_stds, 
                   capsize=5, error_kw={'linewidth': 2}, alpha=0.7)
    plt.ylabel('Position Accuracy')
    plt.title(title)
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, acc, std in zip(bars, accuracies, accuracy_stds):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()


def print_results_summary(results):
    """Print summary table of results."""
    print("\n" + "="*60)
    print("CROSS-ARCHITECTURE EXPERIMENT RESULTS SUMMARY")
    print("="*60)
    print(f"{'Architecture':<15} {'Accuracy':<12} {'Std Dev':<12} {'Hit Rate':<12}")
    print("-"*60)
    
    for name, acc, std, hit_rate in zip(results['arch_names'], results['accuracies'], 
                                       results['accuracy_stds'], results['hit_rates']):
        print(f"{name:<15} {acc:.3f}        {std:.3f}        {hit_rate:.3f}")
    print("="*60)


def main():
    """Main function to run cross-architecture experiments."""
    print("Gram Matrix Decoder - Cross-Architecture Experiments")
    print("=" * 60)
    print("Testing MNIST Dropout Networks: [50,50], [25,25], [100]")
    print("Each architecture will be used as reference against all others")
    
    all_results = run_all_cross_architecture_experiments()
    print_cross_architecture_summary(all_results)
    
    return all_results


if __name__ == "__main__":
    main()