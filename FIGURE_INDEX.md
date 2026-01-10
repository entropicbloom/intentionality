# Figure Index

This document maps figures from the paper to their source scripts and provides instructions for regeneration.

## Quick Reference

| Paper Figure | Description | Script/Source | Output Location |
|--------------|-------------|---------------|-----------------|
| Fig 1 | Decoder validation accuracy (training curves) | WANDB | `images/valid_acc.png` |
| Fig 2 | MNIST model validation accuracies | `decoder/underlying_mnist_performance_real.py` | `underlying/plots/underlying_mnist_performance_comparison_real.png` |
| Fig 3 | Permutation distances | `gram_matrix_decoder/analysis.py` | `images/perm_distances_*.png` |
| Fig 4 | Target-similarity-only output neurons | `decoder/tgt_sim_only_output_neurons_bar_chart.py` | `decoder/tgt_sim_only_output_class_id_accuracy.png` |
| Fig 5 | Neuron ablation study | `gram_matrix_decoder/plot_ablation_results.py` | `gram_matrix_decoder/runs/gram_neuron_ablation_plot.png` |
| Fig 6 | Cross-architecture heatmap | `gram_matrix_decoder/runs/cross_architecture_experiment.py` | `cross_architecture_heatmap_accuracy.png` |
| Fig 7 | Input neuron distance prediction | WANDB / `decoder/r2_comparison_bar_chart_with_error_bars.py` | `decoder/plots/input_neuron_distance_r2_comparison.png` |
| Fig 8 | Target-similarity-only input pixels | WANDB / `decoder/tgt_sim_only_input_neurons_bar_chart.py` | `decoder/tgt_sim_only_input_neuron_distance.png` |
| Appendix | k-NN kernel similarity vs decoder accuracy | `decoder/analysis.ipynb` | N/A |

---

## Detailed Regeneration Instructions

### Figure 1: Decoder Validation Accuracy (Training Curves)

**Type:** WANDB screenshot

**Semantic equivalent:** `images/valid_acc.png`

**Bar chart alternative:**
```bash
cd gram_matrix_decoder
python -c "from runs.classid_comparison import run_comparison_experiment; run_comparison_experiment()"
```
**Output:** `gram_matrix_decoder/model_comparison.png`

**Expected values:**
- untrained: ~10% (chance)
- no_dropout: ~25%
- dropout: ~75%

---

### Figure 2: MNIST Model Validation Accuracies

**Script:** `decoder/underlying_mnist_performance_real.py`

**Regenerate:**
```bash
python decoder/underlying_mnist_performance_real.py
```

**Output:** `underlying/plots/underlying_mnist_performance_comparison_real.png`

**Expected values:**
- untrained: ~9%
- no_dropout: ~95%
- dropout: ~94%

**Requirements:** Model checkpoints in `underlying/saved_models/`

---

### Figure 3: Permutation Distances

**Source:** Pre-generated images in `images/`

**Files:**
- `images/perm_distances_dropout.png`
- `images/perm_distances_no_dropout.png`

**Plotting function:** `gram_matrix_decoder/analysis.py:plot_distance_distribution()`

**To regenerate from experiment data:**
```python
from gram_matrix_decoder.analysis import plot_distance_distribution
# distances array from gram matrix experiment
plot_distance_distribution(distances, seed_idx=0)
```

---

### Figure 4: Target-Similarity-Only Output Neurons

**Script:** `decoder/tgt_sim_only_output_neurons_bar_chart.py`

**Regenerate:**
```bash
cd decoder
python tgt_sim_only_output_neurons_bar_chart.py
```

**Output:** `decoder/tgt_sim_only_output_class_id_accuracy.png`

**Expected values:**
- Full (No Dropout): 0.236
- Target Sim Only (No Dropout): 0.190
- Full (Dropout): 0.753
- Target Sim Only (Dropout): 0.369

**Requirements:** CSV data in `data/output-neurons/`

---

### Figure 5: Neuron Ablation Study

**Script:** `gram_matrix_decoder/plot_ablation_results.py`

**Regenerate:**
```bash
cd gram_matrix_decoder
python plot_ablation_results.py
```

**Output:** `gram_matrix_decoder/runs/gram_neuron_ablation_plot.png`

**Note:** Paper uses bar chart format; script produces line chart with same data.

**Expected pattern:** Performance increases from ~1x (2 neurons) to ~10x (10 neurons) relative to random guessing.

---

### Figure 6: Cross-Architecture Heatmap

**Script:** `gram_matrix_decoder/runs/cross_architecture_experiment.py`

**Regenerate:**
```bash
cd gram_matrix_decoder
python -c "from runs.cross_architecture_experiment import run_cross_architecture_experiment; run_cross_architecture_experiment()"
```

**Output:** `cross_architecture_heatmap_accuracy.png`

**Requirements:** Model checkpoints for architectures `[50,50]`, `[25,25]`, `[100]`

---

### Figure 7: Input Neuron Distance Prediction

**Type:** WANDB screenshot (training curves)

**Description:** Two-panel plot showing R² score over training batches + distance-from-center heatmap.

**Bar chart equivalent:**
```bash
cd decoder
python r2_comparison_bar_chart_with_error_bars.py
```

**Output:** `decoder/plots/input_neuron_distance_r2_comparison.png`

**Expected values:**
- untrained: ~0% (R² ≈ 0)
- no_dropout: 0.844
- dropout: 0.695

**Requirements:** CSV data in `data/input-pixels/`

---

### Figure 8: Target-Similarity-Only Input Pixels

**Type:** WANDB screenshot (training curves)

**Bar chart alternative:**
```bash
cd decoder
python tgt_sim_only_input_neurons_bar_chart.py
```

**Output:** `decoder/tgt_sim_only_input_neuron_distance.png`

**Expected values:**
- Dropout Full: 0.695
- Dropout Target Sim Only: 0.304
- No Dropout Full: 0.844
- No Dropout Target Sim Only: 0.692

**Requirements:** CSV data in `data/input-pixels/`

---

### Appendix: k-NN Kernel Similarity vs Decoder Accuracy

**Source:** `decoder/analysis.ipynb`

**Cell location:** Search for `knn_overlap` function

**To regenerate:** Run the relevant cells in the Jupyter notebook that compute k-NN overlap and create the comparison bar chart.

---

## Additional Plots

### Gram Matrix Decoder Position Accuracy Comparison

**Script:** `gram_matrix_decoder/runs/classid_comparison.py`

**Regenerate:**
```bash
cd gram_matrix_decoder
python -c "from runs.classid_comparison import run_comparison_experiment; run_comparison_experiment()"
```

**Output:** `gram_matrix_decoder/model_comparison.png`

**Expected values (gram matrix decoder):**
- untrained: ~12%
- no_dropout: ~38%
- dropout: ~100%

---

### 3D Reference Geometry Visualization

**Scripts:**
- Matplotlib: `gram_matrix_decoder/runs/visualize_reference_geometry.py`
- Plotly (interactive): `gram_matrix_decoder/runs/visualize_reference_geometry_plotly.py`

**Outputs:**
- `reference_gram_3d_{model_name}.png`
- `gram_matrix_3d_{model_name}.html`

---

### Dataset Classification Performance

**Script:** `decoder/dataset_classification_bar_chart.py`

**Output:** `plots/dataset_classification_performance.png`

---

## Configuration

Key configuration file: `gram_matrix_decoder/config.py`

Important settings:
- `EVAL_DATASET_TYPE`: "mnist" or "fashionmnist"
- `REFERENCE_SEEDS`: Seeds used for reference geometry
- `EVAL_SEEDS`: Seeds used for evaluation

---

## Requirements

All scripts require:
1. Model checkpoints in `underlying/saved_models/`
2. CSV data files in `data/` (for some decoder plots)
3. Dependencies from `requirements.txt`
