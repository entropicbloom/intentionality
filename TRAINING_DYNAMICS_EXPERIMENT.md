# Training Dynamics of Intentionality Experiment

## Overview

This experiment investigates **when intentionality emerges during neural network training**. It answers the fundamental question: *Does a neuron's "meaning" appear suddenly or gradually during training?*

## Research Questions

1. **When does intentionality emerge?**
   - Does it appear in random networks (epoch 0)?
   - Does it emerge gradually or suddenly?
   - At what training epoch can we reliably decode neuron identity?

2. **How does intentionality relate to task performance?**
   - Does intentionality emerge before, during, or after task learning?
   - Is there a correlation between decoder accuracy and task accuracy?

3. **Is intentionality fundamental to network function?**
   - If it emerges early (before high task accuracy), it suggests intentionality is fundamental to how networks learn
   - If it emerges late (after task convergence), it suggests intentionality is a byproduct of optimization

## Experiment Design

### Phase 1: Train Underlying Models with Checkpoints
- Train multiple networks (default: 5 seeds) on MNIST/Fashion-MNIST
- Save model checkpoints at epochs: **[0, 1, 2, 3, 5, 10, 20, 50, 100]**
- Evaluate task accuracy at each checkpoint

### Phase 2: Train Decoders on Each Checkpoint
- For each checkpoint epoch:
  - Load the checkpointed weights
  - Train a Set Transformer decoder to predict neuron class identity
  - Measure decoder accuracy (validation)

### Phase 3: Analysis & Visualization
- Plot decoder accuracy vs training epoch
- Plot task accuracy vs training epoch
- Create combined plots showing emergence patterns
- Compute correlation between task and decoder accuracy
- Identify critical epochs where intentionality emerges

## File Structure

```
underlying/
├── train_with_checkpoints.py          # Trains models, saves checkpoints at specific epochs

decoder/
├── underlying_datasets/
│   └── checkpointed_last_layer.py     # Loads checkpointed model weights
├── setup/
│   └── training_dynamics.py           # Setup functions for training dynamics experiment
├── experiments/
│   └── run_training_dynamics.py       # Main experiment runner
└── visualize_training_dynamics.py     # Visualization scripts

data/
└── training-dynamics/                 # Results CSV files

plots/
└── training_dynamics/                 # Generated plots
```

## Usage

### Quick Start (Small Scale Test)

```bash
# Run a small-scale test with 5 underlying models, 3 decoder seeds
python -m decoder.experiments.run_training_dynamics
```

### Custom Configuration

```python
from decoder.experiments.run_training_dynamics import run_full_training_dynamics_experiment

# Run with custom parameters
run_full_training_dynamics_experiment(
    num_underlying_seeds=10,      # More underlying models for robustness
    num_decoder_seeds=5,          # Multiple decoder seeds for statistics
    num_neurons=10,               # Number of output neurons to decode
    checkpoint_epochs=[0, 1, 2, 3, 5, 10, 20, 50, 100],
    train_decoder_per_epoch=True  # Train new decoder per epoch (vs. train once on final)
)
```

### Visualize Results

```bash
# Visualize results from a specific experiment
python decoder/visualize_training_dynamics.py data/training-dynamics/training_dynamics_fully_connected_dropout_mnist_n10.csv
```

## Configuration Options

### Underlying Model Configuration

```python
underlying_config = {
    'model_class_str': 'fully_connected_dropout',  # or 'fully_connected'
    'dataset_class_str': 'mnist',                  # or 'fashionmnist'
    'batch_size': 256,
    'num_epochs': 100,                             # Train for 100 epochs
    'learning_rate': 0.001,
    'num_workers': 4,
    'num_classes': 10,
    'hidden_dim': [50, 50],                        # 2-layer network
    'varying_dim_bounds': None,
    'models_dir': 'saved_models_checkpoints/'
}
```

### Decoder Configuration

```python
decoder_config = {
    'decoder_class': 'TransformerDecoder',         # Set Transformer
    'preprocessing': 'multiply_transpose',         # Use cosine similarity
    'use_target_similarity_only': False,          # Use full similarity matrix
    'num_neurons': 10,                            # Decode all 10 output neurons
}
```

## Output

### Results CSV

Columns:
- `epoch`: Training epoch of the checkpoint
- `train_acc`: Decoder training accuracy
- `valid_acc`: Decoder validation accuracy
- `decoder_seed`: Random seed for decoder
- `task_acc_mean`: Mean task accuracy of underlying models
- `num_neurons`: Number of neurons decoded

### Visualizations

1. **Decoder Accuracy Plot**: Shows how decoder accuracy evolves across training epochs
2. **Task Accuracy Plot**: Shows underlying model task performance over epochs
3. **Combined Emergence Plot**: Overlays decoder and task accuracy to visualize when intentionality emerges
4. **Correlation Plot**: Scatter plot showing relationship between task and decoder accuracy

## Expected Insights

### Scenario 1: Early Emergence (Intentionality is Fundamental)
- **Pattern**: High decoder accuracy even at epoch 0 (random initialization)
- **Interpretation**: Architecture creates inherent meaning; intentionality is structural
- **Implication**: Neuron identity is partly predetermined by network architecture

### Scenario 2: Gradual Emergence (Intentionality is Learned)
- **Pattern**: Decoder accuracy increases gradually with training
- **Interpretation**: Intentionality develops through optimization
- **Implication**: Meaning emerges from the learning process

### Scenario 3: Correlated Emergence (Intentionality ↔ Performance)
- **Pattern**: Decoder accuracy correlates strongly with task accuracy
- **Interpretation**: Intentionality is a byproduct of learning to solve the task
- **Implication**: Networks develop interpretable structure as they learn

### Scenario 4: Early Emergence (Intentionality Precedes Performance)
- **Pattern**: Decoder accuracy high before task accuracy converges
- **Interpretation**: Networks develop functional organization before achieving high performance
- **Implication**: Intentionality is fundamental to the learning mechanism

## Extensions

### Possible Future Experiments

1. **Layer-wise dynamics**: Track intentionality emergence across different layers
2. **Cross-dataset transfer**: Train decoder on epoch 100 of MNIST, test on all epochs of Fashion-MNIST
3. **Different architectures**: Compare fully connected vs. convolutional networks
4. **Entropy reduction**: Compute ARS (Ambiguity-Reduction Score) at each checkpoint
5. **Adversarial perturbations**: How sensitive is intentionality to weight noise at different epochs?

## Computational Requirements

- **Time**: ~2-3 hours for default configuration (5 underlying models, 3 decoder seeds)
- **Storage**: ~500 MB for checkpoints (5 seeds × 9 epochs)
- **Memory**: 4-8 GB RAM, GPU recommended but not required

## Citation

If you use this experiment in your research, please cite:

```
Training Dynamics of Intentionality Experiment
Part of the Intrinsic Intentionality in Neural Networks project
[Your citation information here]
```

## Notes

- Checkpoint files are saved with format: `seed-{seed}_epoch-{epoch}`
- By default, checkpoints are saved in `saved_models_checkpoints/` to avoid conflicts with regular training
- Results are saved in `data/training-dynamics/`
- Plots are saved in `plots/training_dynamics/`
