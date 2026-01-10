"""Analysis and visualization functions for gram matrix decoder results."""

import numpy as np
import matplotlib.pyplot as plt


def calculate_permutation_accuracy(best_permutations, C):
    """Calculate accuracy over best permutations."""
    identity = np.arange(C)
    
    # collect per-permutation accuracies, then average
    perm_accs = [
        (perm == identity).mean()          # fraction of correct positions
        for perms_for_seed in best_permutations
        for perm in perms_for_seed
    ]
    
    acc_mean = np.mean(perm_accs)
    acc_std = np.std(perm_accs)
    
    print(f"\nAverage position-wise accuracy of best permutations: {acc_mean:.4f}")
    return acc_mean, acc_std


def plot_distance_distribution(distances, seed_idx=0, save_path=None):
    """Plot distance distribution for a given seed."""
    sorted_distances = np.sort(distances[seed_idx])

    fig, axs = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [2, 1]})

    main_color = "#26a69a"   # teal
    dot_color = "#0288d1"    # blue
    highlight_color = "#ef5350"  # red/pink for highlight

    # Full plot (left)
    axs[0].plot(
        sorted_distances,
        color=main_color,
        linewidth=2.5,
        alpha=0.92,
    )
    axs[0].set_xlabel("Neuron permutations", fontsize=18, labelpad=10)
    axs[0].set_ylabel("Distance to reference", fontsize=18, labelpad=10)
    axs[0].set_title("All permutations", fontsize=20, pad=12)
    axs[0].grid(alpha=0.18, linestyle="--")
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    axs[0].tick_params(axis="both", which="major", labelsize=14)

    # Zoomed plot (right)
    zoomed_xlim = min(10, len(sorted_distances))
    if zoomed_xlim > 1:
        axs[1].scatter(
            np.arange(1, zoomed_xlim),      # All except index 0
            sorted_distances[1:zoomed_xlim],
            color=dot_color,
            s=80,
            edgecolor="#009688",
            linewidth=1.5,
            zorder=2,
        )
    # Highlight index 0
    axs[1].scatter(
        0,
        sorted_distances[0],
        color=highlight_color,
        s=120,
        edgecolor="black",
        linewidth=2,
        zorder=3,
        label="Permutation 0"
    )
    axs[1].set_xlim(-1, zoomed_xlim)
    axs[1].set_ylim(sorted_distances[:zoomed_xlim].min() - 0.05, sorted_distances[:zoomed_xlim].max() + 0.05)
    axs[1].set_xlabel("Permutation (zoomed)", fontsize=18, labelpad=10)
    axs[1].set_title(f"Zoom: first {zoomed_xlim}", fontsize=20, pad=12)
    axs[1].grid(alpha=0.18, linestyle="--")
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    axs[1].tick_params(axis="both", which="major", labelsize=14)

    plt.tight_layout(pad=1.5)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return fig, axs