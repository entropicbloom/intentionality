import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the CSV file for output class experiments
df = pd.read_csv('../data/output-classes/tgt-sim-only-experiment.csv')

# Group by model class and use_target_similarity_only
model_groups = df.groupby(['model_class_str', 'use_target_similarity_only'])

# Calculate mean and std for each condition
results = []
for (model_class, use_tgt_sim), group in model_groups:
    mean_acc = group['valid_acc'].mean()
    std_acc = group['valid_acc'].std()
    min_acc = group['valid_acc'].min()
    max_acc = group['valid_acc'].max()
    results.append({
        'model_class': model_class,
        'use_target_similarity_only': use_tgt_sim,
        'mean_acc': mean_acc,
        'std_acc': std_acc,
        'min_acc': min_acc,
        'max_acc': max_acc
    })

# Sort results for consistent ordering (Full first, then Target Sim Only)
results = sorted(results, key=lambda x: (x['model_class'], x['use_target_similarity_only']))

# Prepare data for plotting
model_names = []
means = []
mins = []
maxs = []

for result in results:
    model_name = result['model_class'].replace('_', '-')
    tgt_sim_suffix = '-tgt-sim-only' if result['use_target_similarity_only'] else ''
    model_names.append(f"{model_name}{tgt_sim_suffix}")
    means.append(result['mean_acc'])
    mins.append(result['min_acc'])
    maxs.append(result['max_acc'])

# Calculate error bars (distance from mean to min/max)
lower_errors = [mean - min_val for mean, min_val in zip(means, mins)]
upper_errors = [max_val - mean for mean, max_val in zip(means, maxs)]
errors = [lower_errors, upper_errors]

# Create figure and axis (match input neurons style)
plt.figure(figsize=(8, 6))  # Tighter figure for compact layout

# Create grouped bar positions with tighter spacing
bar_width = 0.25  # Tighter bar width
group_gap = 0.05  # Smaller gap between bars within group
middle_separation = 0.08  # Minimal middle separation
x_positions = [0, bar_width + group_gap, 1 + middle_separation, 1 + middle_separation + bar_width + group_gap]

# Colors: two shades for each group
colors = ["#16a085", "#27ae60", "#2980b9", "#8e44ad"]  # Teal, Green for fully-connected; Blue, Purple for fully-connected-dropout

# Create bar chart with error bars
bars = plt.bar(x_positions, means, yerr=errors, 
               color=colors, alpha=0.7, width=bar_width,
               capsize=5, error_kw={'linewidth': 2})

# Customize the plot
plt.ylabel('Validation Accuracy', fontsize=12)
plt.title('Class ID Decoding Performance', fontsize=14, pad=20)
plt.ylim(0, 1.0)
plt.grid(True, alpha=0.3)

# Add value labels on bars
for bar, mean_val, upper_err in zip(bars, means, upper_errors):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + upper_err + 0.02,
            f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')

# Create two-level x-axis labels
# First level: individual bar labels
bar_labels = ['Full', 'Target Sim\nOnly', 'Full', 'Target Sim\nOnly']
plt.xticks(x_positions, bar_labels, fontsize=10)

# Second level: group labels
group_centers = [(x_positions[0] + x_positions[1])/2, (x_positions[2] + x_positions[3])/2]
group_labels = ['Fully Connected', 'Fully Connected\nDropout']

# Add group labels below the individual labels
ax = plt.gca()
for center, label in zip(group_centers, group_labels):
    plt.text(center, -0.08, label, ha='center', va='top', fontweight='bold', 
             fontsize=12)

# Add separating line between groups
line_x = (x_positions[1] + x_positions[2]) / 2
plt.axvline(x=line_x, color='gray', linestyle='--', alpha=0.5, linewidth=1)

# Adjust layout and save
plt.tight_layout()
plt.savefig('output_class_id_accuracy_4_conditions.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary table
print("\n" + "="*70)
print("OUTPUT CLASS ID ACCURACY RESULTS SUMMARY (4 CONDITIONS)")
print("="*70)
print(f"{'Model':<30} {'Accuracy':<12} {'Range':<20}")
print("-"*70)
for name, mean_val, min_val, max_val in zip(model_names, means, mins, maxs):
    print(f"{name:<30} {mean_val:.3f}        [{min_val:.3f}, {max_val:.3f}]")
print("="*70)