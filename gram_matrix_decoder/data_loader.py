"""Data loading functions for gram matrix decoder experiments."""

import numpy as np
import torch
from config import BASEDIR, REFERENCE_MODEL_FMT, EVAL_MODEL_FMT


def load_last_layer(seed: int, model_type: str = "eval") -> np.ndarray:
    """Load last-layer weight matrix (shape: [num_classes, hidden]).
    
    Args:
        seed: Random seed for model selection
        model_type: Either "reference" or "eval" to specify which model format to use
    """
    model_fmt = REFERENCE_MODEL_FMT if model_type == "reference" else EVAL_MODEL_FMT
    path = BASEDIR / model_fmt.format(seed=seed)
    ckpt = torch.load(path, map_location="cpu")
    
    # Find the last layer dynamically by looking for the highest numbered layer
    layer_keys = [k for k in ckpt.keys() if k.startswith("layers.") and k.endswith(".weight")]
    if not layer_keys:
        raise ValueError(f"No layer weights found in checkpoint at {path}")
    
    # Extract layer numbers and find the maximum
    layer_nums = [int(k.split(".")[1]) for k in layer_keys]
    last_layer_num = max(layer_nums)
    last_layer_key = f"layers.{last_layer_num}.weight"
    
    w = ckpt[last_layer_key].cpu().detach().numpy()  # shape [C, H]
    # L2‑normalise rows so cosine similarity == dot product
    w /= np.linalg.norm(w, axis=1, keepdims=True) + 1e-12
    return w


def gram(w: np.ndarray) -> np.ndarray:
    """Return the CxC Gram matrix of row-normalised weights."""
    return w @ w.T                                    # cosine similarities


def frob(A: np.ndarray, B: np.ndarray) -> float:
    """Frobenius-norm distance ‖A-B‖_F."""
    return np.linalg.norm(A - B, ord="fro")