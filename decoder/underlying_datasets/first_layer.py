import os
import math
import random
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl

__all__ = [
    # helper fns (exported so notebooks can import them)
    "coords", "centre", "get_positional_encoding",
    "radial_indices", "scramble_radial", "random_k_indices",
    # dataset / datamodule
    "FirstLayerDataset", "FirstLayerDataModule",
]

# =============================================================================
# Global pixel geometry
# =============================================================================

WIDTH: int = 28
NUM_PIXELS: int = WIDTH * WIDTH
centre: torch.Tensor = torch.tensor([ (WIDTH - 1) / 2.0 ] * 2)  # (13.5, 13.5)

# Pre‑compute (x,y) coordinates for all flat indices – reused everywhere
xs, ys = torch.meshgrid(torch.arange(WIDTH), torch.arange(WIDTH), indexing="ij")
coords: torch.Tensor = torch.stack((xs.flatten(), ys.flatten()), dim=1).float()  # (784, 2)

# =============================================================================
# Stand‑alone helper functions (importable by notebooks)
# =============================================================================

def get_positional_encoding(pixel_idx: int, encoding_type: str) -> torch.Tensor:
    """Return positional encoding for an MNIST pixel."""
    i, j = divmod(pixel_idx, WIDTH)
    if encoding_type == "2d_normalized":
        return torch.tensor([i / (WIDTH - 1), j / (WIDTH - 1)], dtype=torch.float32)
    if encoding_type == "x_normalized":
        return torch.tensor([i / (WIDTH - 1)], dtype=torch.float32)
    if encoding_type == "y_normalized":
        return torch.tensor([j / (WIDTH - 1)], dtype=torch.float32)
    if encoding_type == "dist_center":
        dist = math.hypot(i - centre[0], j - centre[1])
        return torch.tensor([dist / math.hypot(centre[0], centre[1])], dtype=torch.float32)
    raise ValueError(f"Unknown positional encoding type: {encoding_type}")

# -----------------------------------------------------------------------------
# Subset‑selection helpers
# -----------------------------------------------------------------------------

def radial_indices(target_idx: int, thickness: int, /) -> List[int]:
    """Return indices lying on the ray from *centre* through *target_idx*.

    Thickness *w* means we keep all pixels whose perpendicular distance to that
    ray is ≤ w/2  (Manhattan half‑width).
    """
    if thickness < 1 or thickness > WIDTH:
        raise ValueError("thickness must be in [1, 28]")
    v_target = coords[target_idx] - centre
    if torch.allclose(v_target, torch.zeros(2)):
        v_target = torch.tensor([0.0, 1.0])  # arbitrary direction for centre pixel
    v_hat = v_target / torch.norm(v_target)
    half = thickness / 2.0 + 1e-4
    idxs: List[int] = []
    for idx, xy in enumerate(coords):
        v = xy - centre
        proj = torch.dot(v, v_hat)
        if proj < -1e-6:
            continue  # behind centre
        perp = torch.norm(v - proj * v_hat)
        if perp <= half:
            idxs.append(idx)
    return idxs

def scramble_radial(indices: List[int], target_idx: int) -> List[int]:
    """Scramble *indices* by replacing each non‑target idx with another pixel of
    the *same radius* (distance to centre)."""
    # bucket by integer‑rounded radius
    radius_bins: dict[int, List[int]] = {}
    for idx in range(NUM_PIXELS):
        r = int(round(torch.dist(coords[idx], centre).item()))
        radius_bins.setdefault(r, []).append(idx)
    scrambled: List[int] = []
    for idx in indices:
        if idx == target_idx:
            scrambled.append(idx)
            continue
        r = int(round(torch.dist(coords[idx], centre).item()))
        scrambled.append(random.choice(radius_bins[r]))
    return scrambled

def random_k_indices(k: int, target_idx: int) -> List[int]:
    """Return *k* unique pixel indices (including target)."""
    if k < 1 or k > NUM_PIXELS:
        raise ValueError("k must be 1–784")
    all_others = list(range(NUM_PIXELS))
    all_others.remove(target_idx)
    return [target_idx] + random.sample(all_others, k - 1)

# =============================================================================
# Dataset with sub‑graph masking
# =============================================================================

class FirstLayerDataset(Dataset):
    """Build (similarity‑matrix, positional‑label) pairs for one input neuron."""

    def __init__(
        self,
        dataset_path: str,
        positional_encoding_type: str,
        layer_idx: int = 0,
        *,
        subgraph_type: Optional[str] = None,
        subgraph_param: Optional[int] = None,
        use_target_similarity_only: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.layer_key = f"layers.{layer_idx}.weight"
        self.positional_encoding_type = positional_encoding_type
        self.subgraph_type = subgraph_type
        self.subgraph_param = subgraph_param
        self.use_target_similarity_only = use_target_similarity_only

        valid = {None, "random_k", "radial", "scrambled_radial"}
        if subgraph_type not in valid:
            raise ValueError(f"subgraph_type must be one of {valid}")

        # list checkpoints once
        self.model_files: List[str] = [
            f for f in os.listdir(dataset_path)
            if f.startswith("seed-") and f[5:].split(".")[0].isdigit()
        ]
        if not self.model_files:
            raise FileNotFoundError(f"No model files found in {dataset_path}")

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.model_files) * NUM_PIXELS

    def __getitem__(self, idx: int):
        model_idx, pix_idx = divmod(idx, NUM_PIXELS)

        # ------- load checkpoint & first‑layer weights
        state = torch.load(os.path.join(self.dataset_path, self.model_files[model_idx]), map_location="cpu")
        W: torch.Tensor = state[self.layer_key]  # (H, 784)
        if W.shape[1] != NUM_PIXELS:
            raise ValueError("Unexpected input dimension in weights")

        # ------- build permuted weight matrix (target col first)
        target_col = W[:, pix_idx].clone()
        remaining = [i for i in range(NUM_PIXELS) if i != pix_idx]
        permuted = torch.zeros_like(W)
        permuted[:, 0] = target_col
        permuted[:, 1:] = W[:, random.sample(remaining, len(remaining))]

        # cosine similarity matrix
        normed = permuted / (torch.norm(permuted, dim=0, keepdim=True) + 1e-8)
        sim = normed.T @ normed

        # optional masking
        if self.subgraph_type is not None:
            sim = self._mask(sim, pix_idx)

        label = get_positional_encoding(pix_idx, self.positional_encoding_type)
        
        if self.use_target_similarity_only:
            sim_output = sim[0:1, :]
        else:
            sim_output = sim
        
        return sim_output, label

    # ------------------------------------------------------------------
    def _mask(self, sim: torch.Tensor, pix_idx: int) -> torch.Tensor:
        mask = torch.zeros(NUM_PIXELS, dtype=torch.bool)
        mask[0] = True  # always keep target (col/row 0)

        if self.subgraph_type == "random_k":
            k = self._validate_param(self.subgraph_param, 1, NUM_PIXELS, "k")
            mask[random_k_indices(k, pix_idx)] = True
        elif self.subgraph_type in {"radial", "scrambled_radial"}:
            w = self._validate_param(self.subgraph_param, 1, WIDTH, "w")
            idxs = radial_indices(pix_idx, w)
            if self.subgraph_type == "scrambled_radial":
                idxs = scramble_radial(idxs, pix_idx)
            mask[idxs] = True
        else:
            raise RuntimeError("Unsupported subgraph_type")

        # zero rows/cols outside mask
        masked = sim.clone()
        absent = ~mask
        masked[absent, :] = 0
        masked[:, absent] = 0
        return masked

    # ------------------------------------------------------------------
    @staticmethod
    def _validate_param(value: Optional[int], lo: int, hi: int, name: str) -> int:
        if value is None or value < lo or value > hi:
            raise ValueError(f"{name} must be in [{lo}, {hi}]")
        return value

# =============================================================================
# Lightning DataModule
# =============================================================================

class FirstLayerDataModule(pl.LightningDataModule):
    """Convenience wrapper to train/validate with PyTorch‑Lightning."""

    def __init__(
        self,
        dataset_path: str,
        positional_encoding_type: str,
        batch_size: int = 64,
        num_workers: int = 4,
        *,
        subgraph_type: Optional[str] = None,
        subgraph_param: Optional[int] = None,
        use_target_similarity_only: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.positional_encoding_type = positional_encoding_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.subgraph_type = subgraph_type
        self.subgraph_param = subgraph_param
        self.use_target_similarity_only = use_target_similarity_only

    # ------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None):
        ds = FirstLayerDataset(
            self.dataset_path,
            positional_encoding_type=self.positional_encoding_type,
            subgraph_type=self.subgraph_type,
            subgraph_param=self.subgraph_param,
            use_target_similarity_only=self.use_target_similarity_only,
        )
        n_train = int(0.8 * len(ds))
        self.train_set, self.val_set = random_split(ds, [n_train, len(ds) - n_train])

    # ------------------------------------------------------------------
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)
