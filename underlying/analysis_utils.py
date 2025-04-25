import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform

from datasets.MNIST import MNISTDataModule
from pytorch_models.fully_connected import FullyConnected, FullyConnectedDropout, FullyConnectedGenerative

# Class mappings (same as in main.py)
MODEL_MAP = {
    'fully_connected': FullyConnected,
    'fully_connected_dropout': FullyConnectedDropout,
}

def evaluate_model(model, dataloader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Evaluate a model on a dataloader and return accuracy
    """
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            inputs = inputs.view(inputs.size(0), -1)
                
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def test_model(model_type, dataset_type, seed, untrained="", hidden_dim=None, base_model_path='saved_models', data_path="./underlying/data"):
    """
    Load and test a model on the MNIST validation set
    
    Args:
        model_type (str): Type of model (e.g., 'fully_connected')
        dataset_type (str): Type of dataset (e.g., 'mnist')
        seed (int): Seed number
        untrained (str): Empty string or '-untrained' for untrained models
        hidden_dim (list): Hidden dimensions for the model.
        base_model_path (str): Path to the directory containing saved models.
        data_path (str): Path to the dataset.
    
    Returns:
        float: Accuracy percentage on validation set
    """
    # Set up data module
    data_module = MNISTDataModule(batch_size=256, num_workers=4, data_path=data_path)
    data_module.prepare_data()
    data_module.setup(stage='validate')
    
    # Construct model path
    hidden_dim_str = str(hidden_dim).replace(" ", "")
    model_path = os.path.join(base_model_path, f'{model_type}-{dataset_type}{untrained}-hidden_dim_{hidden_dim_str}/seed-{seed}')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at path: {model_path}")
    
    # Create model instance of the correct type
    model_class = MODEL_MAP[model_type]
    input_dim = data_module.input_dim
    model = model_class(num_classes=10, input_dim=input_dim, hidden_dim=hidden_dim)
    
    # Load the state dictionary
    model.load_state_dict(torch.load(model_path))
    
    # Get validation dataloader
    val_dataloader = data_module.val_dataloader()
    
    # Evaluate the model
    accuracy = evaluate_model(model, val_dataloader)
    
    return accuracy

def idx2xy(i, width=28):
  """Converts flat index i (row-major) to (x, y) coords."""
  return (i % width, i // width)

def display_pixel_map(vector, width=28, **imshow_kwargs):
    """Displays a flat vector as a 2D image map."""
    # 1. Create an empty 2D array (height, width)
    pixel_map = np.zeros((width, width), dtype=float)

    # 2. Iterate through the flat vector
    vector_np = vector.cpu().numpy() if isinstance(vector, torch.Tensor) else np.asarray(vector)
    for i, value in enumerate(vector_np):
        # Get the (x, y) coordinates for the flat index i
        x, y = idx2xy(i, width=width)
        # Assign the value to the map using [row, column] indexing
        pixel_map[y, x] = value

    # 3. Display the reconstructed map
    plt.imshow(pixel_map, interpolation='nearest', **imshow_kwargs)
    plt.colorbar()

def create_lr_gradient_vector(w=28):
  """Generates a vector for a left-to-right gradient (0.0=left, 1.0=right)."""
  # Create a grid where the value is just the x-coordinate
  # 'indexing='xy'' ensures x varies along columns (left-to-right)
  x, _ = np.meshgrid(np.arange(w), np.arange(w), indexing='xy')

  # Normalize x-coordinate to be between 0.0 and 1.0
  # Add epsilon for the case w=1 to avoid division by zero
  norm_factor = float(w - 1) + 1e-9
  pixel_map_2d = x / norm_factor

  # Ensure values are clipped just in case and flatten to 1D vector
  return np.clip(pixel_map_2d, 0.0, 1.0).flatten()

def radial_similarity_curve(weights_layer0, img_size=28, bins=None, device='cpu'):
    """
    Compute ⟨cosθ⟩ as a function of pixel-grid distance d for one network.

    Args
    ----
    weights_layer0 : torch.Tensor  shape (hidden_dim, img_size*img_size)
        Column j contains the outgoing weights of input neuron j.
    img_size       : int          width/height of the input grid (MNIST = 28).
    bins           : 1-D arraylike of distances at which to sample the curve.
                     Default = [0, 1, 2, …, img_size-1].
    device         : 'cpu' or 'cuda'.

    Returns
    -------
    d_vals : np.ndarray  (len(bins),)
    sim    : np.ndarray  (len(bins),)  average cosine similarity at each d.
    """
    W = weights_layer0.to(device)                       # (H, 784)
    W_norm = W / (W.norm(dim=0, keepdim=True) + 1e-9)   # column‑wise ℓ2 normalisation
    cos = torch.matmul(W_norm.t(), W_norm).cpu().numpy()# (784, 784)

    # pre‑compute all pairwise Euclidean distances between pixel coordinates
    coords = np.array([(i, j) for i in range(img_size) for j in range(img_size)])
    dists  = squareform(pdist(coords, metric='euclidean'))

    if bins is None:
        bins = np.arange(img_size)        # 0, 1, …, 27 pixels
    bins = np.asarray(bins)

    sim = np.empty_like(bins, dtype=float)
    for k, d in enumerate(bins):
        mask = (np.round(dists) == d)     # integer‑distance shell
        # Handle cases where a distance bin might be empty
        sim[k] = cos[mask].mean() if mask.any() else np.nan
    return bins, sim 