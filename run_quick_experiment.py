"""
Quick experimental run with minimal resources to generate results for visualization.
This runs a small-scale version of the training dynamics experiment.
"""
import os
import sys

# Add paths
sys.path.insert(0, os.path.dirname(__file__))

from decoder.experiments.run_training_dynamics import run_full_training_dynamics_experiment

# Run a very small-scale experiment for quick results
print("=" * 80)
print("RUNNING QUICK TRAINING DYNAMICS EXPERIMENT")
print("=" * 80)
print("\nThis will:")
print("  - Train 3 underlying models (seeds 0-2)")
print("  - Save checkpoints at epochs: 0, 1, 2, 3, 5")
print("  - Train 2 decoders per checkpoint")
print("  - Decode 10 output neurons")
print("  - Take approximately 15-20 minutes")
print("=" * 80)
print()

underlying_config = {
    'model_class_str': 'fully_connected',  # No dropout for faster training
    'dataset_class_str': 'mnist',
    'batch_size': 512,  # Larger batch = faster
    'num_epochs': 5,    # Fewer epochs for quick run
    'learning_rate': 0.001,
    'num_workers': 0,   # Avoid multiprocessing issues
    'num_classes': 10,
    'hidden_dim': [50, 50],
    'varying_dim_bounds': None,
    'models_dir': 'saved_models_checkpoints/'
}

# Run experiment
results_file = run_full_training_dynamics_experiment(
    num_underlying_seeds=3,     # Just 3 models
    num_decoder_seeds=2,        # 2 decoder seeds
    num_neurons=10,             # All 10 output neurons
    underlying_config=underlying_config,
    checkpoint_epochs=[0, 1, 2, 3, 5],  # Fewer checkpoints
    train_decoder_per_epoch=True
)

print("\n" + "=" * 80)
print("EXPERIMENT COMPLETE!")
print("=" * 80)
print(f"\nResults saved to: {results_file}")
print("\nNow generating visualizations...")
print("=" * 80)

# Generate visualizations
from decoder.visualize_training_dynamics import plot_training_dynamics, create_summary_table

plot_training_dynamics(results_file)
create_summary_table(results_file)

print("\n" + "=" * 80)
print("ALL DONE! Check the plots/ directory for visualizations.")
print("=" * 80)
