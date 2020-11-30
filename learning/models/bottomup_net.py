import torch
from torch import nn
from torch.nn import functional as F

class BottomUpNet(nn.Module):
    def __init__(self, n_in, n_hidden, share_weights=True, max_blocks=5):
        """ This network is given input of size (N, K, n_in) where N, K can vary per batch.
        :param n_in: Number of block-specific parameters.
        :param n_hidden: Number of hidden units unsed throughout the network.
        """
        super(BottomUpNet, self).__init__()
        self.aggregate = nn.Parameter(torch.zeros(1, n_hidden))    

        if share_weights:
            M = self.build_M(n_in, n_hidden)
            O = self.build_O(n_in, n_hidden)
            self.M = nn.ModuleList([M for _ in range(max_blocks)])
            self.O = nn.ModuleList([O for _ in range(max_blocks)])
        else:
            self.M = nn.ModuleList([self.build_M(n_in, n_hidden) for _ in range(max_blocks)])
            self.O = nn.ModuleList([self.build_O(n_in, n_hidden) for _ in range(max_blocks)])

        self.n_in, self.n_hidden = n_in, n_hidden
        self.share_weights = share_weights

    def build_M(self, n_in, n_hidden):
        return nn.Sequential(nn.Linear(n_in+n_hidden, n_hidden),
                            nn.ReLU(),
                            nn.Linear(n_hidden, n_hidden),
                            nn.ReLU(),
                            nn.Linear(n_hidden, n_hidden),
                            nn.ReLU())

    def build_O(self, n_in, n_hidden):
        return nn.Sequential(nn.Linear(n_in+n_hidden, n_hidden),            
                            nn.ReLU(),
                            nn.Linear(n_hidden, n_hidden),
                            nn.ReLU(),
                            nn.Linear(n_hidden, 1))

    def forward(self, towers):
        """
        :param towers: (N, K, n_in) tensor describing the tower.
        :param k: Number of times to iterate the graph update.
        """
        N, K, _ = towers.shape

        bottom_summary = self.aggregate.expand(N, self.n_hidden)
        preds = []
        for kx in range(K):
            # Calculate edge updates for each node: (N, K, n_hidden) 
            x = torch.cat([bottom_summary, towers[:, kx, :]], dim=1)
            pred = torch.sigmoid(self.O[kx](x))
            preds.append(pred)

            bottom_summary = self.M[kx](x)
        
        preds = torch.cat(preds, dim=1)
        return preds.prod(dim=1).unsqueeze(-1)

            
        
