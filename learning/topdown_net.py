import torch
from torch import nn
from torch.nn import functional as F

class TopDownNet(nn.Module):
    def __init__(self, n_in, n_hidden):
        """ This network is given input of size (N, K, n_in) where N, K can vary per batch.
        :param n_in: Number of block-specific parameters.
        :param n_hidden: Number of hidden units unsed throughout the network.
        """
        super(TopDownNet, self).__init__()
        self.aggregate = nn.Parameter(torch.zeros(1, n_hidden))    

        self.M = nn.Sequential(nn.Linear(n_in+n_hidden, n_hidden),
                               nn.ReLU(),
                               nn.Linear(n_hidden, n_hidden),
                               nn.ReLU(),
                               nn.Linear(n_hidden, n_hidden),
                               nn.ReLU())

        self.O = nn.Sequential(nn.Linear(n_in+n_hidden, n_hidden),            
                               nn.ReLU(),
                               nn.Linear(n_hidden, n_hidden),
                               nn.ReLU(),
                               nn.Linear(n_hidden, 1))

        self.n_in, self.n_hidden = n_in, n_hidden

    def forward(self, towers, k):
        """
        :param towers: (N, K, n_in) tensor describing the tower.
        :param k: Number of times to iterate the graph update.
        """
        N, K, _ = towers.shape
        towers = torch.flip(towers, dims=[1])

        top_summary = self.aggregate.expand(N, self.n_hidden)
        preds = []
        for kx in range(K):
            # Calculate edge updates for each node: (N, K, n_hidden) 
            x = torch.cat([top_summary, towers[:, kx, :]], dim=1)
            pred = torch.sigmoid(self.O(x))
            preds.append(pred)

            top_summary = self.M(x)
        
        preds = torch.cat(preds, dim=1)
        return preds.prod(dim=1)

        
        
