"""Bar chart for self-attention decoder validation accuracy comparison.

This script creates a bar chart showing the final validation accuracy
for the self-attention decoder across different training paradigms.
Values are from the original decoder experiments (Figure 1 in paper).
"""

import matplotlib.pyplot as plt
import numpy as np

# Self-attention decoder final validation accuracies (from paper Figure 1)
# These are the converged values after training
model_names = ['Untrained', 'No Dropout', 'Dropout']
means = [0.10, 0.25, 0.75]  # Final validation accuracies
stds = [0.01, 0.02, 0.03]   # Approximate standard deviations

# Create figure
plt.figure(figsize=(10, 6))

# Blue-green color palette (consistent with other plots)
colors = ["#2980b9", "#16a085", "#8e44ad"]  # Blue, Teal, Purple

# Create bar chart with error bars
bars = plt.bar(model_names, means, yerr=stds,
               color=colors, alpha=0.7,
               capsize=6, error_kw={'linewidth': 2.5})

# Customize the plot (paper-ready font sizes)
plt.ylabel('Validation Accuracy', fontsize=18)
plt.title('Self-Attention Decoder Performance', fontsize=20, pad=20)
plt.ylim(0, 1.0)
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=16)
plt.yticks(fontsize=14)

# Add horizontal line at 0.1 for chance accuracy
plt.axhline(y=0.1, color='gray', linestyle='--', alpha=0.6, linewidth=1.5, label='Chance (0.1)')

# Add value labels on bars
for bar, mean_val, std_val in zip(bars, means, stds):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_val + 0.02,
            f'{mean_val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=14)

# Adjust layout and save
plt.tight_layout()
plt.savefig('plots/self_attention_decoder_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary
print("\n" + "="*50)
print("SELF-ATTENTION DECODER RESULTS SUMMARY")
print("="*50)
print(f"{'Model':<15} {'Accuracy':<12} {'Std Dev':<12}")
print("-"*50)
for name, mean_val, std_val in zip(model_names, means, stds):
    print(f"{name:<15} {mean_val:.2f}         {std_val:.2f}")
print("="*50)
