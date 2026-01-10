import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

# Load the CSV files for each condition
dropout_df = pd.read_csv('../data/input-pixels/dropout.csv')
no_dropout_df = pd.read_csv('../data/input-pixels/no-dropout.csv')
dropout_tgt_sim_df = pd.read_csv('../data/input-pixels/dropout-tgt-sim-only.csv')
no_dropout_tgt_sim_df = pd.read_csv('../data/input-pixels/no-dropout-tgt-sim-only.csv')

# Calculate distances from center (same as in the notebook)
distances_from_center = []
for i in range(28):
    for j in range(28):
        dist = math.sqrt((i - 13.5)**2 + (j - 13.5)**2)
        max_dist = math.sqrt(13.5**2 + 13.5**2)
        distances_from_center.append(dist/max_dist)

label_variance = np.var(distances_from_center)

# Get final values (last row) for each condition
def get_final_r2_stats(df, label_variance):
    final_row = df.iloc[-1]
    mean_mse = final_row.iloc[1]  # Second column is mean MSE
    min_mse = final_row.iloc[2]   # Third column is min MSE
    max_mse = final_row.iloc[3]   # Fourth column is max MSE
    
    # Convert MSE to R²
    mean_r2 = 1.0 - (mean_mse / label_variance)
    min_r2 = 1.0 - (max_mse / label_variance)  # Note: max MSE gives min R²
    max_r2 = 1.0 - (min_mse / label_variance)  # Note: min MSE gives max R²
    
    return mean_r2, min_r2, max_r2

# Get R² statistics for each condition
dropout_mean, dropout_min, dropout_max = get_final_r2_stats(dropout_df, label_variance)
no_dropout_mean, no_dropout_min, no_dropout_max = get_final_r2_stats(no_dropout_df, label_variance)
dropout_tgt_sim_mean, dropout_tgt_sim_min, dropout_tgt_sim_max = get_final_r2_stats(dropout_tgt_sim_df, label_variance)
no_dropout_tgt_sim_mean, no_dropout_tgt_sim_min, no_dropout_tgt_sim_max = get_final_r2_stats(no_dropout_tgt_sim_df, label_variance)

# Prepare data for plotting (group dropout and no-dropout together)
model_names = ['dropout', 'dropout-tgt-sim-only', 'no-dropout', 'no-dropout-tgt-sim-only']
means = [dropout_mean, dropout_tgt_sim_mean, no_dropout_mean, no_dropout_tgt_sim_mean]
mins = [dropout_min, dropout_tgt_sim_min, no_dropout_min, no_dropout_tgt_sim_min]
maxs = [dropout_max, dropout_tgt_sim_max, no_dropout_max, no_dropout_tgt_sim_max]

# Calculate error bars (distance from mean to min/max)
lower_errors = [mean - min_val for mean, min_val in zip(means, mins)]
upper_errors = [max_val - mean for mean, max_val in zip(means, maxs)]
errors = [lower_errors, upper_errors]

# Create figure and axis (match gram matrix style)
plt.figure(figsize=(10, 7))  # Slightly larger for better readability

# Create grouped bar positions with tighter spacing
bar_width = 0.25  # Tighter bar width
group_gap = 0.05  # Smaller gap between bars within group
middle_separation = 0.08  # Minimal middle separation
x_positions = [0, bar_width + group_gap, 1 + middle_separation, 1 + middle_separation + bar_width + group_gap]

# Colors: two shades for each group
colors = ["#16a085", "#27ae60", "#2980b9", "#8e44ad"]  # Teal, Green for dropout; Blue, Purple for no-dropout

# Create bar chart with error bars (match gram matrix style)
bars = plt.bar(x_positions, means, yerr=errors,
               color=colors, alpha=0.7, width=bar_width,
               capsize=6, error_kw={'linewidth': 2.5})

# Customize the plot (paper-ready font sizes)
plt.ylabel('R² Score', fontsize=18)
plt.title('Distance-from-Center Decoding Performance', fontsize=20, pad=20)
plt.ylim(0, 1.0)  # Start from 0 since all values are positive
plt.grid(True, alpha=0.3)
plt.yticks(fontsize=14)

# Add value labels on bars (match gram matrix style)
for bar, mean_val, upper_err in zip(bars, means, upper_errors):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + upper_err + 0.02,
            f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=14)

# Create two-level x-axis labels
# First level: individual bar labels
bar_labels = ['Full', 'Target Sim\nOnly', 'Full', 'Target Sim\nOnly']
plt.xticks(x_positions, bar_labels, fontsize=14)

# Second level: group labels
group_centers = [(x_positions[0] + x_positions[1])/2, (x_positions[2] + x_positions[3])/2]
group_labels = ['Dropout', 'No Dropout']

# Add group labels below the individual labels (using data coordinates for better alignment)
ax = plt.gca()
for center, label in zip(group_centers, group_labels):
    plt.text(center, -0.12, label, ha='center', va='top', fontweight='bold',
             fontsize=16)

# Add separating line between groups
line_x = (x_positions[1] + x_positions[2]) / 2
plt.axvline(x=line_x, color='gray', linestyle='--', alpha=0.5, linewidth=1)

# Adjust layout and save (match gram matrix style)
plt.tight_layout()
plt.savefig('tgt_sim_only_input_neuron_distance.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary table (match gram matrix style)
print("\n" + "="*65)
print("INPUT NEURON DISTANCE R² RESULTS SUMMARY (4 CONDITIONS)")
print("="*65)
print(f"{'Model':<25} {'R² Score':<12} {'Range':<20}")
print("-"*65)
for name, mean_val, min_val, max_val in zip(model_names, means, mins, maxs):
    print(f"{name:<25} {mean_val:.3f}        [{min_val:.3f}, {max_val:.3f}]")
print("="*65)