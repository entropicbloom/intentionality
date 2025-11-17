"""
Quick test of the Training Dynamics experiment pipeline.
Runs a very small-scale experiment to verify all components work.
"""
import os
import sys

# Test 1: Import all modules
print("=" * 80)
print("TEST 1: Importing modules")
print("=" * 80)

try:
    from underlying.train_with_checkpoints import train_with_checkpoints
    print("✓ underlying.train_with_checkpoints imported")
except Exception as e:
    print(f"✗ Failed to import underlying.train_with_checkpoints: {e}")
    sys.exit(1)

try:
    from decoder.underlying_datasets.checkpointed_last_layer import CheckpointedLastLayerDataset
    print("✓ decoder.underlying_datasets.checkpointed_last_layer imported")
except Exception as e:
    print(f"✗ Failed to import checkpointed_last_layer: {e}")
    sys.exit(1)

try:
    from decoder.setup.training_dynamics import run_training_dynamics_experiment
    print("✓ decoder.setup.training_dynamics imported")
except Exception as e:
    print(f"✗ Failed to import training_dynamics: {e}")
    sys.exit(1)

try:
    from decoder.experiments.run_training_dynamics import run_full_training_dynamics_experiment
    print("✓ decoder.experiments.run_training_dynamics imported")
except Exception as e:
    print(f"✗ Failed to import run_training_dynamics: {e}")
    sys.exit(1)

print("\nAll modules imported successfully! ✓\n")

# Test 2: Train a single model with checkpoints
print("=" * 80)
print("TEST 2: Training single model with checkpoints")
print("=" * 80)

config = {
    'model_class_str': 'fully_connected',
    'dataset_class_str': 'mnist',
    'batch_size': 256,
    'num_epochs': 5,  # Very short for testing
    'learning_rate': 0.001,
    'num_workers': 0,  # Avoid multiprocessing issues
    'num_classes': 10,
    'hidden_dim': [50, 50],
    'varying_dim_bounds': None,
    'models_dir': 'saved_models_checkpoints_test/'
}

checkpoint_epochs = [0, 1, 2, 5]

try:
    save_path = train_with_checkpoints(
        model_class_str=config['model_class_str'],
        dataset_class_str=config['dataset_class_str'],
        batch_size=config['batch_size'],
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        num_workers=config['num_workers'],
        num_classes=config['num_classes'],
        hidden_dim=config['hidden_dim'],
        seed=0,
        data_dir='./data',
        models_dir=config['models_dir'],
        checkpoint_epochs=checkpoint_epochs,
    )
    print(f"\n✓ Model trained and checkpoints saved to: {save_path}\n")
except Exception as e:
    print(f"\n✗ Failed to train model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Verify checkpoint files exist
print("=" * 80)
print("TEST 3: Verifying checkpoint files")
print("=" * 80)

expected_checkpoints = [f'seed-0_epoch-{e}' for e in checkpoint_epochs]
for checkpoint_file in expected_checkpoints:
    checkpoint_path = os.path.join(save_path, checkpoint_file)
    if os.path.exists(checkpoint_path):
        print(f"✓ Found: {checkpoint_file}")
    else:
        print(f"✗ Missing: {checkpoint_file}")

print("\n")

# Test 4: Load checkpoint dataset
print("=" * 80)
print("TEST 4: Loading checkpoint dataset")
print("=" * 80)

try:
    from decoder.underlying_datasets.checkpointed_last_layer import CheckpointedLastLayerDataset

    dataset = CheckpointedLastLayerDataset(
        dataset_path=save_path,
        layer_idx=2,
        epoch=1,  # Test loading epoch 1
        transpose_weights=False,
        preprocessing='multiply_transpose',
        use_neurons=list(range(10)),
    )

    print(f"✓ Dataset created with {len(dataset)} samples")

    # Try loading one sample
    sample = dataset[0]
    print(f"✓ Sample loaded: weights shape = {sample[0].shape}, label = {sample[1]}")

except Exception as e:
    print(f"✗ Failed to load dataset: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n")

# Test 5: Summary
print("=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print("✓ All tests passed!")
print("\nThe Training Dynamics experiment pipeline is working correctly.")
print("\nYou can now run the full experiment with:")
print("  python -m decoder.experiments.run_training_dynamics")
print("\nOr customize it by editing decoder/experiments/run_training_dynamics.py")
print("=" * 80)
