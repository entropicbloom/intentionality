# Expected Results: Training Dynamics of Intentionality

## What The Experiment Will Show

Once the dependencies are installed, running the experiment will reveal **when intentionality emerges** during neural network training. Here's what we expect to discover:

## Hypothesis & Expected Patterns

### Most Likely Scenario: **Gradual Emergence Correlated with Task Learning**

Based on the existing research in this codebase, I predict we'll see:

#### 1. **Epoch 0 (Random Initialization)**
- **Decoder Accuracy**: ~10-15% (slightly above random chance of 10%)
- **Task Accuracy**: ~10% (random)
- **Interpretation**: Some weak structural intentionality exists from initialization, possibly due to:
  - Symmetry breaking from random weights
  - Network architecture imposing constraints
  - Statistical properties of weight initialization

#### 2. **Epochs 1-3 (Early Training)**
- **Decoder Accuracy**: Rising to ~30-50%
- **Task Accuracy**: Rising to ~60-80%
- **Interpretation**: Rapid emergence of intentionality as networks begin to specialize
- **Key Finding**: Decoder accuracy may lag slightly behind task accuracy

#### 3. **Epochs 5-10 (Mid Training)**
- **Decoder Accuracy**: ~60-80%
- **Task Accuracy**: ~90-95%
- **Interpretation**: Strong intentionality develops; neurons have clear functional roles
- **Pattern**: Both curves begin to plateau

#### 4. **Epochs 20-100 (Late Training)**
- **Decoder Accuracy**: ~85-95%
- **Task Accuracy**: ~95-98%
- **Interpretation**: Intentionality stabilizes and possibly sharpens further
- **Observation**: Diminishing returns on both metrics

## Visualizations We'll Generate

### 1. Decoder Accuracy Over Time
```
100% |                                          ████████
     |                                    ██████
 80% |                              ██████
     |                        ██████
 60% |                  ██████
     |            ██████
 40% |      ██████
     |  ████
 20% |██
 10% |█_____________________________________________
      0    1    2    3    5   10   20   50   100
                    Training Epoch
```

### 2. Combined Emergence Plot
Shows both decoder and task accuracy overlaid, revealing:
- **If they rise together**: Intentionality is a byproduct of task learning
- **If decoder lags**: Intentionality emerges after networks learn the task
- **If decoder leads**: Intentionality is necessary for task learning (most interesting!)

### 3. Correlation Plot
Scatter plot of task accuracy vs decoder accuracy for all checkpoint/seed combinations:
- **Strong positive correlation (r > 0.9)**: Intentionality tightly coupled to performance
- **Weak correlation (r < 0.5)**: Intentionality is somewhat independent of task success
- **Nonlinear relationship**: Complex interaction between the two

## Alternative Scenarios & Their Implications

### Scenario A: **Strong Early Intentionality**
If decoder accuracy is already 40-50% at epoch 0:
- **Implication**: Architecture creates significant inherent meaning
- **Biological Parallel**: Like how biological neurons have predispositions
- **Follow-up**: Compare different architectures (ConvNets vs MLPs)

### Scenario B: **Sudden Emergence**
If decoder accuracy jumps sharply at a specific epoch:
- **Implication**: Phase transition in weight space
- **Connection**: Related to neural network "grokking" phenomenon
- **Follow-up**: Investigate loss landscape around that epoch

### Scenario C: **Decoder Precedes Task**
If decoder accuracy exceeds task accuracy early on:
- **Implication**: Networks develop interpretable structure before solving the task
- **Significance**: Intentionality is fundamental to learning mechanism
- **Impact**: Most exciting result for the field!

### Scenario D: **No Early Intentionality**
If decoder accuracy stays near 10% until later epochs:
- **Implication**: Intentionality is purely learned, not structural
- **Contrast**: Would differ from existing "lottery ticket" hypothesis
- **Follow-up**: Test on pruned networks

## Quantitative Metrics We'll Compute

From the CSV results:

1. **Emergence Epoch**: When decoder accuracy first exceeds 50%
2. **Correlation Coefficient**: Between task and decoder accuracy
3. **Relative Improvement**: `(acc_100 - acc_0) / acc_0` for both metrics
4. **Plateau Epoch**: When improvement drops below 5% per epoch
5. **Lead/Lag**: Which metric reaches milestones first

## Comparison to Existing Experiments

This codebase already shows:
- Trained networks have ~90% decoder accuracy (from existing experiments)
- Untrained networks have ~20-30% decoder accuracy
- Dropout affects intentionality patterns

**Our experiment will fill in the gap**: showing the continuous trajectory from untrained → trained!

## Files That Will Be Generated

```
data/training-dynamics/
└── training_dynamics_fully_connected_mnist_n10.csv

plots/training_dynamics/
├── decoder_accuracy_fully_connected_mnist_n10.png
├── task_accuracy_fully_connected_mnist_n10.png
├── emergence_combined_fully_connected_mnist_n10.png
├── correlation_fully_connected_mnist_n10.png
└── training_dynamics_fully_connected_mnist_n10_summary.csv

saved_models_checkpoints/
└── fully_connected-mnist-epochs-5-hidden-[50, 50]-varying-False/
    ├── seed-0_epoch-0
    ├── seed-0_epoch-1
    ├── seed-0_epoch-2
    ├── seed-0_epoch-3
    ├── seed-0_epoch-5
    └── ... (for seeds 1-2)
```

## Next Steps After Running

Once we see the results:

1. **If intentionality emerges early**: Test whether it predicts future performance
2. **If gradual emergence**: Model it with sigmoid/exponential curves
3. **If correlated with task**: Compute exact mathematical relationship
4. **Compare architectures**: Run same experiment on dropout vs no-dropout
5. **Cross-dataset**: Does MNIST emergence differ from Fashion-MNIST?

## How to Run

Once dependencies install:

```bash
# Quick version (~15-20 minutes)
python run_quick_experiment.py

# Full version (~2-3 hours)
python -m decoder.experiments.run_training_dynamics
```

Then view the gorgeous plots in `plots/training_dynamics/`! 📊✨
