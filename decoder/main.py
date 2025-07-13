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
    run_mixed_hidden_dims_classid,
    run_random_k_subgraph_inputpixels
)

# Runner function definitions - MOVED to decoder/runners.py

if __name__ == '__main__':
    run_random_k_subgraph_inputpixels(k_values=[4, 8, 16, 32, 64, 128, 256, 512])