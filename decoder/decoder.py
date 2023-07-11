# from https://github.com/juho-lee/set_transformer

from modules import *


class DeepSet(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, dim_hidden=128):
        super(DeepSet, self).__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.enc = nn.Sequential(
                nn.Linear(dim_input, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden))
        self.dec = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, num_outputs*dim_output))

    def forward(self, X):
        X = self.enc(X).mean(-2)
        X = self.dec(X).reshape(-1, self.num_outputs, self.dim_output)
        return X

class Transformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(Transformer, self).__init__()
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                #ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)
                #SAB(dim_input, dim_hidden, num_heads, ln=ln)

        )
                
        self.dec = nn.Sequential(
                #PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                #SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output)
        )

    def forward(self, X):
        # multiply X matrix by transpose
        X = X @ X.transpose(1,2)
        

        X = self.enc(X)

        output =  self.dec(X[:,0,:])

        return output
    
class FCDecoder(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, dim_hidden=128, num_inds=None, num_heads=None, ln=False):
        super(FCDecoder, self).__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.layers = nn.Sequential(
                nn.Linear(dim_input, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
        )

    def forward(self, X):
        X = X @ X.transpose(1,2)
        return self.layers(X[:,0,:])