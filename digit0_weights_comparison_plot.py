import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def load_digit_weights(model_type, dataset_type, seed, digit, hidden_dim=[50, 50]):
    """Load the incoming weights for a specific digit output neuron from a trained model"""
    hidden_dim_str = str(hidden_dim).replace(" ", "")
    model_path = f'underlying/saved_models/{model_type}-{dataset_type}-hidden_dim_{hidden_dim_str}/seed-{seed}'

    model_state = torch.load(model_path)
    # Get the final layer weights (output layer) - shape should be [10, hidden_dim]
    output_weights = model_state['layers.2.weight'].cpu().detach()

    # Extract weights for the specified digit
    digit_weights = output_weights[digit, :]  # Shape: [hidden_dim]

    return digit_weights

def main():
    model_type = 'fully_connected'
    dataset_type = 'mnist'
    hidden_dim = [50, 50]

    # Collect weight vectors for digits 0 and 1 across 50 seeds
    num_seeds = 50
    digits = [0, 1]

    weight_vectors = []
    labels = []  # Will store (seed, digit) pairs

    print("Loading weight vectors...")
    for seed in range(num_seeds):
        for digit in digits:
            try:
                weights = load_digit_weights(model_type, dataset_type, seed, digit, hidden_dim)
                weight_vectors.append(weights.numpy())
                labels.append((seed, digit))
                print(f"Loaded seed {seed}, digit {digit}")
            except Exception as e:
                print(f"Failed to load seed {seed}, digit {digit}: {e}")

    if not weight_vectors:
        print("No weight vectors loaded!")
        return

    # Convert to numpy array for dimensionality reduction
    weight_matrix = np.array(weight_vectors)  # Shape: [num_vectors, weight_dim]
    print(f"Weight matrix shape: {weight_matrix.shape}")

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)
    weights_2d = pca.fit_transform(weight_matrix)
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

    # Create scatter plot with larger fonts for slides
    plt.figure(figsize=(12, 10))
    plt.rcParams.update({'font.size': 18})

    # Define colors for each digit class
    digit_colors = {0: 'blue', 1: 'red'}

    # Plot points, grouped by digit
    for digit in digits:
        digit_indices = [i for i, (seed, d) in enumerate(labels) if d == digit]
        digit_coords = weights_2d[digit_indices]

        # Create scatter plot for this digit
        scatter = plt.scatter(digit_coords[:, 0], digit_coords[:, 1],
                            c=digit_colors[digit],
                            alpha=0.7,
                            s=120,
                            label=f'Digit {digit}',
                            edgecolors='black',
                            linewidth=0.5)

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=20)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=20)
    plt.title('Weight Vector PCA', fontsize=24)
    plt.legend(fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=16)

    plt.tight_layout()
    plt.savefig('digit_weights_pca_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\\nPlot saved:")
    print("- digit_weights_pca_scatter.png")
    print(f"Total vectors plotted: {len(weight_vectors)}")
    print(f"Digits 0 (circles) and 1 (squares), colored by seed")

if __name__ == "__main__":
    main()