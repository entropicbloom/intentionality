"""Visualize reference gram matrix geometry in 3D using PCA."""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from experiment import build_reference_geometry


def visualize_reference_gram_matrix_3d(model_config=None):
    """Visualize the reference gram matrix in 3D using PCA."""
    if model_config is None:
        # Use trained no-dropout model by default
        model_config = {
            'name': 'dropout',
            'ref_model_type': 'fully_connected_dropout',
            'ref_untrained': False
        }
    
    print(f"Visualizing reference gram matrix for {model_config['name']} model...")
    
    # Build model format string
    ref_untrained_suffix = "-untrained" if model_config['ref_untrained'] else ""
    ref_model_fmt = f"{model_config['ref_model_type']}-{config.REFERENCE_DATASET_TYPE}{ref_untrained_suffix}-hidden_dim_{config.REFERENCE_HIDDEN_DIM}/seed-{{seed}}"
    
    print(f"Model format: {ref_model_fmt}")
    
    # Build reference geometry
    G_ref, C = build_reference_geometry(
        reference_seeds=config.REFERENCE_SEEDS,
        reference_model_fmt=ref_model_fmt,
        basedir=config.BASEDIR,
        all_perms=config.ALL_PERMS
    )
    
    print(f"Reference gram matrix shape: {G_ref.shape}")
    print(f"Number of classes: {C}")
    
    # Perform eigendecomposition (PCA on gram matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(G_ref)
    
    # Sort by eigenvalues (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Project to 3D using top 3 eigenvectors
    coords_3d = eigenvectors[:, :3] * np.sqrt(eigenvalues[:3])
    
    # Create 3D visualization
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the 10 class neurons
    colors = plt.cm.tab10(np.linspace(0, 1, C))
    for i in range(C):
        ax.scatter(coords_3d[i, 0], coords_3d[i, 1], coords_3d[i, 2], 
                  c=[colors[i]], s=100, alpha=0.8, label=f'Class {i}')
        # Add digit label next to each point
        ax.text(coords_3d[i, 0], coords_3d[i, 1], coords_3d[i, 2], f'  {i}', 
                color=colors[i], fontsize=12, fontweight='bold')
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(False)
    
    # Remove all axes, labels, and panes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('') 
    ax.set_zlabel('')
    ax.set_title('')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    # Hide axis lines
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    # Make axis spines invisible
    ax.xaxis._axinfo["axisline"]["color"] = (1, 1, 1, 0)
    ax.yaxis._axinfo["axisline"]["color"] = (1, 1, 1, 0)
    ax.zaxis._axinfo["axisline"]["color"] = (1, 1, 1, 0)
    
    # Print explained variance
    total_var = np.sum(eigenvalues)
    explained_var_3d = np.sum(eigenvalues[:3]) / total_var
    print(f"\nExplained variance by top 3 components: {explained_var_3d:.3f}")
    print(f"Individual component ratios: {eigenvalues[:3] / total_var}")
    print(f"All eigenvalues: {eigenvalues}")
    
    plt.tight_layout()
    output_file = Path(__file__).parent / f'reference_gram_3d_{model_config["name"]}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_file}")
    plt.show()
    
    return coords_3d, eigenvalues, eigenvectors


if __name__ == "__main__":
    print("Reference Gram Matrix 3D Visualization")
    print("="*50)
    
    # Single model visualization
    print("\n1. Visualizing trained dropout model geometry...")
    visualize_reference_gram_matrix_3d()