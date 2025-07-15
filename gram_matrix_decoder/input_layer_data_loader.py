"""Data loading functions for input layer gram matrix decoder experiments."""

import numpy as np
import torch
from config import BASEDIR, REFERENCE_MODEL_FMT, EVAL_MODEL_FMT
import sys
sys.path.append('..')
from decoder.underlying_datasets.first_layer import get_positional_encoding, WIDTH


def load_first_layer(seed: int, model_type: str = "eval", model_fmt: str = None, basedir=None) -> np.ndarray:
    """Load first-layer weight matrix (shape: [hidden, input_pixels]).
    
    Args:
        seed: Random seed for model selection
        model_type: Either "reference" or "eval" to specify which model format to use
        model_fmt: Custom model format string, overrides default based on model_type
        basedir: Custom base directory, overrides default BASEDIR
    """
    if model_fmt is None:
        model_fmt = REFERENCE_MODEL_FMT if model_type == "reference" else EVAL_MODEL_FMT
    if basedir is None:
        basedir = BASEDIR
    
    path = basedir / model_fmt.format(seed=seed)
    ckpt = torch.load(path, map_location="cpu")
    
    # Get first layer weights
    first_layer_key = "layers.0.weight"
    if first_layer_key not in ckpt:
        raise ValueError(f"No first layer weights found in checkpoint at {path}")
    
    w = ckpt[first_layer_key].cpu().detach().numpy()  # shape [H, 784]
    # L2-normalise columns (each input neuron) so cosine similarity == dot product
    w /= np.linalg.norm(w, axis=0, keepdims=True) + 1e-12
    return w


def gram_input(w: np.ndarray) -> np.ndarray:
    """Return the input_pixels x input_pixels Gram matrix of column-normalised weights."""
    return w.T @ w  # cosine similarities between input neurons


def get_distance_labels() -> np.ndarray:
    """Get distance-from-center labels for all input pixels."""
    labels = []
    for pixel_idx in range(WIDTH * WIDTH):
        dist_label = get_positional_encoding(pixel_idx, "dist_center")
        labels.append(dist_label.item())
    return np.array(labels)


def frob(A: np.ndarray, B: np.ndarray) -> float:
    """Frobenius-norm distance ‖A-B‖_F."""
    # More efficient than np.linalg.norm for Frobenius norm
    diff = A - B
    return np.sqrt(np.sum(diff * diff))