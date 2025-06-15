import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


# Import runner functions (now named experiment functions)
from experiments import (
    run_ablation_experiments_classid,
    run_main_experiments_classid,
    run_main_experiments_inputpixels,
    run_inputpixels_subsets,
    run_similarity_comparison_classid,
    run_mixed_hidden_dims_classid
)

# Runner function definitions - MOVED to decoder/runners.py

if __name__ == '__main__':
    # Example usage with custom architectures
    run_mixed_hidden_dims_classid(
        num_seeds=5,
        train_hidden_dim=[50, 50],
        valid_hidden_dim=[100],
        train_varying_dim=True,
        train_samples=8000,
        valid_samples=2000
    )
