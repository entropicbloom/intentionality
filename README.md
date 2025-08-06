# Investigating Intrinsic Intentionality in Neural Networks

This project investigates whether neural networks develop inherent, discoverable meaning in their weight structures - what philosophers might call "intrinsic intentionality". Specifically, I examine whether the functional role of neurons can be determined solely from their weight patterns, without requiring knowledge of that specific network's training history (i.e. how it was hooked up to inputs and outputs).

## Research Question

Can we determine what a neuron "means" (its functional role) just by looking at its weights and their relationships to other neurons? If so, this suggests that neural networks develop consistent, interpretable internal structures that transcend individual training runs.

We investigate two specific cases:
- **Output neuron class identity**: Can we determine which class an output neuron represents?
- **Input neuron pixel position**: Can we determine which pixel position an input neuron corresponds to?

## Installation

```bash
pip install -r requirements.txt
```

## Methodology

### Training Base Networks
- Multiple neural networks are trained on classification tasks
- Networks vary in architecture (number of layers, hidden dimensions: 25-100 neurons) and random seeds
- Each network develops its own unique weights through training

### Experiments

1. **Output Layer Experiment**
   - Extract output layer weights from trained networks
   - Shuffle the neurons (rows) randomly
   - Predict which neuron corresponds to which output class
   - Uses both Set Transformer and Gram Matrix decoders
   - Tests whether functional role is unambiguously encoded in weight patterns

2. **Input Layer Experiment**
   - Extract input layer weights from trained networks
   - Shuffle the neurons randomly
   - Decode the distance from center of input pixels based on weight patterns
   - Uses Set Transformer decoder
   - Tests whether spatial relationships in input space are unambiguously encoded in weight structures

### Decoding Approaches

1. **Set Transformer Decoder**
   - Permutation-invariant neural network architecture
   - Can handle variable numbers of neurons
   - Generalizes across different random seeds and architectures
   - Used for both input and output layer experiments

2. **Gram Matrix Decoder**
   - Compute averaged cosine similarity matrix for reference networks
   - For each evaluation network, test all permutations of neurons
   - Find which permutation produces cosine similarity matrix closest to reference
   - Tests whether networks share consistent geometric structure in weight space
   - Currently only applied to output layer experiments due to tractability constraints

## Project Structure

- **`underlying/`**: Base neural network training and model implementations
- **`decoder/`**: Set Transformer decoder experiments for weight pattern analysis
- **`gram_matrix_decoder/`**: Alternative decoding approach using Gram matrix representations
- **`data/`**: Experimental results and CSV outputs
- **`images/`**: Generated plots and visualizations

## Results

For detailed experimental results and analysis, please see the full report at: https://entropicbloom.github.io/consciousness/
