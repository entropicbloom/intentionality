"""
Visualization scripts for Training Dynamics of Intentionality experiment.

Creates plots showing:
1. Decoder accuracy vs training epoch
2. Task accuracy vs training epoch
3. Combined plot showing emergence of intentionality relative to task learning
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys


def plot_training_dynamics(csv_path, save_dir='plots/training_dynamics'):
    """
    Create comprehensive visualizations of training dynamics.

    Args:
        csv_path (str): Path to the results CSV file
        save_dir (str): Directory to save plots
    """
    # Load data
    df = pd.read_csv(csv_path)

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Set style
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)

    # Extract configuration from filename for plot titles
    filename = os.path.basename(csv_path)
    config_str = filename.replace('training_dynamics_', '').replace('.csv', '')

    # 1. Decoder accuracy over epochs
    print("Creating decoder accuracy plot...")
    plot_decoder_accuracy(df, config_str, save_dir)

    # 2. Task accuracy over epochs
    print("Creating task accuracy plot...")
    plot_task_accuracy(df, config_str, save_dir)

    # 3. Combined plot: decoder accuracy vs task accuracy
    print("Creating combined emergence plot...")
    plot_emergence_combined(df, config_str, save_dir)

    # 4. Correlation plot: decoder accuracy vs task accuracy
    print("Creating correlation plot...")
    plot_correlation(df, config_str, save_dir)

    print(f"\nAll plots saved to: {save_dir}")


def plot_decoder_accuracy(df, config_str, save_dir):
    """Plot decoder accuracy over training epochs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Aggregate across decoder seeds
    summary = df.groupby('epoch').agg({
        'valid_acc': ['mean', 'std'],
        'train_acc': ['mean', 'std']
    }).reset_index()

    epochs = summary['epoch']
    valid_mean = summary['valid_acc']['mean']
    valid_std = summary['valid_acc']['std']
    train_mean = summary['train_acc']['mean']
    train_std = summary['train_acc']['std']

    # Plot validation accuracy
    ax.plot(epochs, valid_mean, 'o-', linewidth=2, markersize=8,
            label='Validation Accuracy', color='#2E86AB')
    ax.fill_between(epochs, valid_mean - valid_std, valid_mean + valid_std,
                     alpha=0.3, color='#2E86AB')

    # Plot training accuracy
    ax.plot(epochs, train_mean, 's--', linewidth=2, markersize=8,
            label='Training Accuracy', color='#A23B72', alpha=0.7)
    ax.fill_between(epochs, train_mean - train_std, train_mean + train_std,
                     alpha=0.2, color='#A23B72')

    ax.set_xlabel('Training Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Decoder Accuracy', fontsize=14, fontweight='bold')
    ax.set_title(f'Decoder Performance Across Training\n{config_str}',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # Add baseline (random chance = 0.1 for 10 classes)
    ax.axhline(y=0.1, color='gray', linestyle=':', linewidth=2, label='Random Chance', alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'decoder_accuracy_{config_str}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_task_accuracy(df, config_str, save_dir):
    """Plot task accuracy of underlying models over training epochs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Aggregate task accuracy across epochs
    summary = df.groupby('epoch').agg({
        'task_acc_mean': ['mean', 'std']
    }).reset_index()

    epochs = summary['epoch']
    task_mean = summary['task_acc_mean']['mean']
    task_std = summary['task_acc_mean']['std']

    # Plot
    ax.plot(epochs, task_mean, 'o-', linewidth=3, markersize=10,
            label='Task Accuracy', color='#F18F01')
    ax.fill_between(epochs, task_mean - task_std, task_mean + task_std,
                     alpha=0.3, color='#F18F01')

    ax.set_xlabel('Training Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Classification Accuracy', fontsize=14, fontweight='bold')
    ax.set_title(f'Underlying Model Task Performance\n{config_str}',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # Add baseline
    ax.axhline(y=0.1, color='gray', linestyle=':', linewidth=2, label='Random Chance', alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'task_accuracy_{config_str}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_emergence_combined(df, config_str, save_dir):
    """
    Plot decoder and task accuracy on the same graph to visualize
    when intentionality emerges relative to task learning.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Aggregate data
    summary = df.groupby('epoch').agg({
        'valid_acc': ['mean', 'std'],
        'task_acc_mean': ['mean', 'std']
    }).reset_index()

    epochs = summary['epoch']
    decoder_mean = summary['valid_acc']['mean']
    decoder_std = summary['valid_acc']['std']
    task_mean = summary['task_acc_mean']['mean']
    task_std = summary['task_acc_mean']['std']

    # Plot decoder accuracy
    ax.plot(epochs, decoder_mean, 'o-', linewidth=3, markersize=10,
            label='Decoder Accuracy (Intentionality)', color='#2E86AB')
    ax.fill_between(epochs, decoder_mean - decoder_std, decoder_mean + decoder_std,
                     alpha=0.3, color='#2E86AB')

    # Plot task accuracy
    ax.plot(epochs, task_mean, 's-', linewidth=3, markersize=10,
            label='Task Accuracy (Performance)', color='#F18F01')
    ax.fill_between(epochs, task_mean - task_std, task_mean + task_std,
                     alpha=0.3, color='#F18F01')

    ax.set_xlabel('Training Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title(f'Emergence of Intentionality vs Task Learning\n{config_str}',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # Add baseline
    ax.axhline(y=0.1, color='gray', linestyle=':', linewidth=2, alpha=0.5)

    # Add annotations for key points
    if len(epochs) > 0:
        # Find when decoder accuracy exceeds 0.5
        idx_decoder_50 = np.where(decoder_mean >= 0.5)[0]
        if len(idx_decoder_50) > 0:
            epoch_decoder_50 = epochs.iloc[idx_decoder_50[0]]
            ax.axvline(x=epoch_decoder_50, color='#2E86AB', linestyle='--',
                      alpha=0.5, linewidth=2)
            ax.text(epoch_decoder_50, 0.05, f'Decoder > 0.5\n(epoch {epoch_decoder_50})',
                   fontsize=10, ha='center', color='#2E86AB', fontweight='bold')

        # Find when task accuracy exceeds 0.9
        idx_task_90 = np.where(task_mean >= 0.9)[0]
        if len(idx_task_90) > 0:
            epoch_task_90 = epochs.iloc[idx_task_90[0]]
            ax.axvline(x=epoch_task_90, color='#F18F01', linestyle='--',
                      alpha=0.5, linewidth=2)
            ax.text(epoch_task_90, 0.95, f'Task > 0.9\n(epoch {epoch_task_90})',
                   fontsize=10, ha='center', color='#F18F01', fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'emergence_combined_{config_str}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_correlation(df, config_str, save_dir):
    """
    Plot correlation between decoder accuracy and task accuracy.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Use all data points (not aggregated)
    task_acc = df['task_acc_mean']
    decoder_acc = df['valid_acc']
    epochs = df['epoch']

    # Scatter plot with color-coded epochs
    scatter = ax.scatter(task_acc, decoder_acc, c=epochs, cmap='viridis',
                        s=100, alpha=0.6, edgecolors='black', linewidth=1)

    # Add colorbar for epochs
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Training Epoch', fontsize=12, fontweight='bold')

    # Fit linear regression
    z = np.polyfit(task_acc, decoder_acc, 1)
    p = np.poly1d(z)
    x_line = np.linspace(task_acc.min(), task_acc.max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.8, label='Linear Fit')

    # Compute correlation
    correlation = np.corrcoef(task_acc, decoder_acc)[0, 1]

    ax.set_xlabel('Task Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylabel('Decoder Accuracy', fontsize=14, fontweight='bold')
    ax.set_title(f'Correlation: Task vs Decoder Accuracy\n{config_str}\n(r = {correlation:.3f})',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])

    # Add diagonal line (y=x)
    ax.plot([0, 1], [0, 1], 'k:', linewidth=2, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'correlation_{config_str}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def create_summary_table(csv_path, save_dir='plots/training_dynamics'):
    """
    Create a summary table of key metrics.

    Args:
        csv_path (str): Path to the results CSV file
        save_dir (str): Directory to save the table
    """
    df = pd.read_csv(csv_path)

    # Aggregate by epoch
    summary = df.groupby('epoch').agg({
        'valid_acc': ['mean', 'std'],
        'train_acc': ['mean', 'std'],
        'task_acc_mean': ['mean', 'std']
    }).reset_index()

    summary.columns = ['Epoch', 'Decoder Valid Mean', 'Decoder Valid Std',
                      'Decoder Train Mean', 'Decoder Train Std',
                      'Task Acc Mean', 'Task Acc Std']

    # Round to 3 decimal places
    summary = summary.round(3)

    # Save as CSV
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.basename(csv_path).replace('.csv', '_summary.csv')
    save_path = os.path.join(save_dir, filename)
    summary.to_csv(save_path, index=False)

    print(f"\nSummary table saved to: {save_path}")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == '__main__':
    # Check if CSV path is provided as argument
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Use default path (update this based on your experiment output)
        csv_path = 'data/training-dynamics/training_dynamics_fully_connected_dropout_mnist_n10.csv'

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        print("Usage: python visualize_training_dynamics.py <path_to_csv>")
        sys.exit(1)

    print(f"Visualizing training dynamics from: {csv_path}\n")
    plot_training_dynamics(csv_path)
    create_summary_table(csv_path)
    print("\nVisualization complete!")
