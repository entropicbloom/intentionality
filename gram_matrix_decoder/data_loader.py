"""Data loading functions for gram matrix decoder experiments."""

import numpy as np
import torch
from config import BASEDIR, MODEL_FMT


def load_last_layer(seed: int) -> np.ndarray:
    """Load last-layer weight matrix (shape: [num_classes, hidden])."""
    path = BASEDIR / MODEL_FMT.format(seed=seed)
    ckpt = torch.load(path, map_location="cpu")
    w = ckpt["layers.2.weight"].cpu().detach().numpy()  # shape [C, H]
    # L2‑normalise rows so cosine similarity == dot product
    w /= np.linalg.norm(w, axis=1, keepdims=True) + 1e-12
    return w


def gram(w: np.ndarray) -> np.ndarray:
    """Return the CxC Gram matrix of row-normalised weights."""
    return w @ w.T                                    # cosine similarities


def frob(A: np.ndarray, B: np.ndarray) -> float:
    """Frobenius-norm distance ‖A-B‖_F."""
    return np.linalg.norm(A - B, ord="fro")