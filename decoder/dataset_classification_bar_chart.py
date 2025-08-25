#!/usr/bin/env python3
"""
Create a bar chart for dataset classification performance.

This script creates a bar chart showing the performance of different model types
in classifying whether networks were trained on MNIST vs Fashion-MNIST.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    # Read the CSV data
    data_path = "../data/dataset-classification/dataset_classification.csv"
    df = pd.read_csv(data_path)
    
    # Group by model type and calculate statistics
    results = {}
    for model_type in df['model_class_str'].unique():
        model_data = df[df['model_class_str'] == model_type]['valid_acc'].values
        results[model_type] = model_data
    
    # Calculate means and standard deviations
    model_names = list(results.keys())
    means = [np.mean(results[model]) for model in model_names]
    stds = [np.std(results[model]) for model in model_names]
    
    # Create the plot - thinner and more elegant
    plt.figure(figsize=(6, 6))
    
    # Use the same colors as previous gram matrix plots
    colors = ["#2980b9", "#16a085", "#8e44ad"]  # Blue, Teal, Purple
    
    # Create bar chart with error bars - thinner bars
    bars = plt.bar(model_names, means, yerr=stds, 
                   color=colors[:len(model_names)], alpha=0.7, 
                   capsize=5, error_kw={'linewidth': 2}, width=0.5)
    
    # Customize the plot
    plt.title('Dataset Classification Performance\n(MNIST vs Fashion-MNIST)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Validation Accuracy', fontsize=12)
    plt.xlabel('Model Type', fontsize=12)
    
    # Set y-axis limits to show the full range
    plt.ylim(0, 1.0)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{mean:.3f}±{std:.3f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Clean up model names for display - simplify to just show dropout status
    simplified_labels = []
    for name in model_names:
        if 'dropout' in name:
            simplified_labels.append('Dropout')
        else:
            simplified_labels.append('No Dropout')
    
    ax = plt.gca()
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(simplified_labels)
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_path = "plots/dataset_classification_performance.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Dataset classification bar chart saved to {output_path}")

if __name__ == "__main__":
    main()