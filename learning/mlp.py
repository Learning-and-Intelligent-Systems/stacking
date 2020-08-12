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
            nn.Linear(n_hidden, 1),
        )

    def forward(self, towers, k=None):
        N, _, _ = towers.shape
        x = towers.view(N, -1)
        x = self.W(x)
        return torch.sigmoid(x)[...,0]


