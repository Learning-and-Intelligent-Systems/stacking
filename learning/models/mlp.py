import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    """
    Implements a MLP for a fixed number of blocks.
    """

    def __init__(self, n_blocks, n_hidden):
        super(MLP, self).__init__()
        self.n_in = n_blocks*14

        # the node feature update weights
        self.W = nn.Sequential(
            nn.Linear(self.n_in, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, towers, k=None):
        N, _, _ = towers.shape
        x = towers.view(N, -1)
        x = self.W(x)
        return torch.sigmoid(x)[...,0]


class FeedForward(nn.Module):
    """ general purpose feedforward NN

    Extends:
        nn.Module
    """
    def __init__(self, d_in, d_out, h_dims=[32, 32]):
        super(FeedForward, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.h_dims = h_dims
        all_dims = [d_in] + list(h_dims) + [d_out]

        # create a linear layer and nonlinearity for each hidden dim
        modules = []
        for i in range(len(all_dims) - 1):
            modules.append(nn.Linear(all_dims[i], all_dims[i+1]))
            modules.append(nn.LeakyReLU())

        modules.pop(-1)  # don't include the last nonlinearity
        self.layers = nn.Sequential(*modules)  # add modules to net

    def forward(self, *xs):
        x = torch.cat(xs, axis=1)
        return self.layers(x)