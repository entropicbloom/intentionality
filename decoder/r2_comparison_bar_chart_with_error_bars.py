import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

# Load the CSV files for each condition
dropout_df = pd.read_csv('../data/input-pixels/dropout.csv')
no_dropout_df = pd.read_csv('../data/input-pixels/no-dropout.csv') 
untrained_df = pd.read_csv('../data/input-pixels/untrained.csv')

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
untrained_mean, untrained_min, untrained_max = get_final_r2_stats(untrained_df, label_variance)

# Prepare data for plotting (match gram matrix order and names)
model_names = ['untrained', 'no_dropout', 'dropout']  # Match gram matrix order
means = [untrained_mean, no_dropout_mean, dropout_mean]  # Reorder to match
mins = [untrained_min, no_dropout_min, dropout_min]
maxs = [untrained_max, no_dropout_max, dropout_max]

# Calculate error bars (distance from mean to min/max)  
lower_errors = [mean - min_val for mean, min_val in zip(means, mins)]
upper_errors = [max_val - mean for mean, max_val in zip(means, maxs)]
errors = [lower_errors, upper_errors]

# Create figure and axis (match gram matrix style)
plt.figure(figsize=(10, 6))

# Blue-green color palette to match gram matrix plots
colors = ["#2980b9", "#16a085", "#8e44ad"]  # Blue, Teal, Purple (same as gram matrix)

# Create bar chart with error bars (match gram matrix style)
bars = plt.bar(model_names, means, yerr=errors, 
               color=colors, alpha=0.7, 
               capsize=5, error_kw={'linewidth': 2})

# Customize the plot (match gram matrix style exactly)
plt.ylabel('R² Score', fontsize=12)
plt.title('Input Neuron Distance to Center Decoding R² Score Comparison', fontsize=14, pad=20)
plt.ylim(-0.1, 1.0)
plt.grid(True, alpha=0.3)

# Add value labels on bars (match gram matrix style)
for bar, mean_val, upper_err in zip(bars, means, upper_errors):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + upper_err + 0.02,
            f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')

# Add horizontal line at y=0 for reference
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)

# Adjust layout and save (match gram matrix style)
plt.tight_layout()
plt.savefig('input_neuron_distance_r2_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary table (match gram matrix style)
print("\n" + "="*50)
print("INPUT NEURON DISTANCE R² RESULTS SUMMARY")
print("="*50)
print(f"{'Model':<15} {'R² Score':<12} {'Range':<20}")
print("-"*50)
for name, mean_val, min_val, max_val in zip(model_names, means, mins, maxs):
    print(f"{name:<15} {mean_val:.3f}        [{min_val:.3f}, {max_val:.3f}]")
print("="*50)