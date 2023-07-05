import torch
import torch.nn as nn
import torch.nn.functional as F


# Regular PyTorch Module
class FullyConnected(nn.Module):
    def __init__(self, num_classes, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = num_classes
        self.hidden_dim = hidden_dim

        current_dim = input_dim
        self.layers = nn.ModuleList()
        for hdim in hidden_dim:
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, num_classes))

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x 

