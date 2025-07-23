# Investigating Intrinsic Intentionality in Neural Networks

This project investigates whether neural networks develop inherent, discoverable meaning in their weight structures - what philosophers might call "intrinsic intentionality". Specifically, I examine whether the role of output neurons (which digit they classify) can be determined solely from their weight patterns, without requiring knowledge of that specific network's training history.

## Research Question

Can we determine what a neuron "means" (which digit it classifies) just by looking at its weights and their relationships to other neurons? If so, this suggests that neural networks develop consistent, interpretable internal structures that transcend individual training runs.

## Methodology

1. **Training Base Networks**
   - Multiple neural networks are trained on MNIST classification
   - Networks vary in architecture (hidden dimensions: 25-100 neurons)
   - Each network develops its own unique weights through training

2. **Decoding Experiment**
   - Extract output layer weights from trained networks
   - Shuffle the neurons (rows) randomly
   - Use a Set Transformer to predict which neuron corresponds to which digit
   - Crucially: The decoder generalizes across different random seeds of base networks, as well as across different architectures.

## Project Structure

- `underlying/`: Base neural networks trained on MNIST
- `decoder/`: Set Transformer for weight pattern analysis

## Results

The experiments demonstrate that neural networks do develop discoverable patterns in their output layer weights that indicate their functional role. I tested this hypothesis across three training paradigms:

1. **Untrained Networks (Control)**
   - Random weights, no training
   - Decoder accuracy: ~10% (chance level)
   - Confirms decoder isn't exploiting spurious patterns

2. **Standard Training**
   - Normal backpropagation
   - Decoder accuracy: ~25%
   - Shows emergence of meaningful weight patterns

3. **Training with Dropout**
   - Backpropagation with dropout
   - Decoder accuracy: ~75%
   - Significant improvement in decodability

### Key Findings

- Output neurons develop consistent, decodable patterns in their connectivity
- Dropout dramatically improves decodability (25% → 75%)

## Interpretation

The superior performance with dropout likely occurs because:
- Dropout encourages neurons to rely on population activity rather than single-neuron pathways
- Output neurons representing similar digits share more features in their input weights
- This creates more consistent and detectable patterns in the relational weight structure

These results suggest that even simple MNIST-predicting ANNs structurally encode representations, making them decodable without knowing anything about how the network was linked to the 'outside world' (i.e. input pixels and output classes). Please consider reading [my article](https://entropicbloom.github.io/consciousness/) for more details and to learn about possible implications in philosophy of mind.
