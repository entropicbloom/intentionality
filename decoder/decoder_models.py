# from https://github.com/juho-lee/set_transformer

from modules import *

class TransformerDecoder(nn.Module):
    def __init__(
        self, 
        dim_input: int, 
        num_outputs: int, 
        dim_output: int,
        num_inds: int = 32, 
        dim_hidden: int = 128, 
        num_heads: int = 4, 
        ln: bool = False,
        encoder_components: list = ['sab', 'sab'],  # List of encoder components
        decoder_components: list = ['linear']  # List of decoder components
    ):
        super().__init__()
        self.dim_input = dim_input
        
        # Build encoder sequence
        encoder_layers = []
        curr_dim = dim_input
        for comp in encoder_components:
            if comp == 'sab':
                encoder_layers.append(SAB(curr_dim, dim_hidden, num_heads, ln=ln))
                curr_dim = dim_hidden
            elif comp == 'isab':
                encoder_layers.append(ISAB(curr_dim, dim_hidden, num_heads, num_inds, ln=ln))
                curr_dim = dim_hidden
            else:
                raise ValueError(f"Unknown encoder component: {comp}")
        self.enc = nn.Sequential(*encoder_layers)
        
        # Build decoder sequence
        decoder_layers = []
        curr_dim = dim_hidden
        for comp in decoder_components:
            if comp == 'linear':
                decoder_layers.append(nn.Linear(curr_dim, dim_output))
                curr_dim = dim_output
            elif comp == 'pma':
                decoder_layers.append(PMA(curr_dim, num_heads, num_outputs, ln=ln))
            elif comp == 'sab':
                decoder_layers.append(SAB(curr_dim, curr_dim, num_heads, ln=ln))
            else:
                raise ValueError(f"Unknown decoder component: {comp}")
        self.dec = nn.Sequential(*decoder_layers)

    # dimensionality of X is (batch_size, num_inputs, dim_inputs)
    def forward(self, X):
        
        # If input is 2D (batch, seq_len) and dim_input is 1, reshape to (batch, seq_len, 1)
        if X.ndim == 2 and self.dim_input == 1:
            X = X.unsqueeze(-1)
        # If input is 2D and represents a feature vector, reshape to (batch, 1, features)
        elif X.ndim == 2 and X.shape[1] == self.dim_input:
            X = X.unsqueeze(1)

        X = self.enc(X)
        return self.dec(X[:,0,:])

class FCDecoder(nn.Module):
    """
    Only looks at the first neuron.
    """
    def __init__(
        self, 
        dim_input: int, 
        num_outputs: int, 
        dim_output: int, 
        dim_hidden: int = 128, 
        num_inds: int = None, 
        num_heads: int = None, 
        ln: bool = False
    ):
        super().__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        
        self.layers = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, X):
        return self.layers(X[:, 0, :])