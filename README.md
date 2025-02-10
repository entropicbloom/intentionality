# Investigating Intrinsic Intentionality in Neural Networks

This project investigates whether neural networks develop inherent, discoverable meaning in their weight structures - what philosophers might call "intrinsic intentionality". Specifically, we examine whether the role of output neurons (which digit they classify) can be determined solely from their weight patterns, without requiring knowledge of that specific network's training history.

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
   - Crucially: The decoder generalizes across different random seeds
   
3. **Key Insight**
   If our decoder can successfully identify digit-neurons across different networks, this suggests that:
   - Networks develop consistent, recognizable patterns for each digit
   - These patterns are inherent to the task, not just artifacts of specific training runs
   - There might be a "natural" way that networks organize information about digits

## Project Structure

- `underlying/`: Base neural networks trained on MNIST
- `decoder/`: Set Transformer for weight pattern analysis

## Results

Our experiments demonstrate that neural networks do develop discoverable patterns in their output layer weights that indicate their functional role. We tested this hypothesis across three training paradigms:

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
- Results suggest networks organize information in interpretable ways

### Interpretation

The superior performance with dropout likely occurs because:
- Dropout encourages neurons to rely on population activity rather than single-neuron pathways
- Output neurons representing similar digits share more features in their input weights
- This creates more consistent and detectable patterns in the weight structure

These results support the existence of intrinsic intentionality in neural networks, as we can decode neuron function across different random seeds without any knowledge of the specific network's training history.

## Conclusion