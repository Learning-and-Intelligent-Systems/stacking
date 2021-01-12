import torch
from torch import nn
from torch.nn import functional as F

class FCGN(nn.Module):
    def __init__(self, n_in, n_hidden):
        """ This network is given input of size (N, K, n_in) where N, K can vary per batch.
        :param n_in: Number of block-specific parameters.
        :param n_hidden: Number of hidden units unsed throughout the network.
        """
        super(FCGN, self).__init__()

        # Note (Mike): When tuning, keep this shallow as it helps to compare 
        # the untransformed features for the edges.
        self.E = nn.Sequential(nn.Linear(n_in, n_hidden))
                               #nn.ReLU())

        # Message function that compute relation between two nodes and outputs a message vector.
        self.M = nn.Sequential(nn.Linear(2*n_in, n_hidden),
                               nn.ReLU(),
                               nn.Linear(n_hidden, n_hidden),
                               nn.ReLU(),
                               nn.Linear(n_hidden, n_hidden),
                               nn.ReLU())

        # Update function that updates a node based on the sum of its messages.
        self.U = nn.Sequential(nn.Linear(n_in+n_hidden, n_hidden),            
                               #nn.ReLU(),
                               #nn.Linear(n_hidden, n_hidden),
                               nn.ReLU())

        # Output function that predicts stability.
        self.O = nn.Sequential(nn.Linear(n_hidden, n_hidden),
                               nn.ReLU(),
                               #nn.Linear(n_hidden, n_hidden),
                               #nn.ReLU(),
                               nn.Linear(n_hidden, 1))
        
        self.n_in, self.n_hidden = n_in, n_hidden
        self.init = nn.Parameter(torch.zeros(1, 1, n_hidden))

    def edge_fn(self, towers, h):
        N, K, _ = towers.shape

        # Get features between all node. 
        # xx.shape = (N, K, K, 2*n_in)
        x = towers 
        x = x[:, :, None, :].expand(N, K, K, self.n_in)
        xx = torch.cat([x, x.transpose(1, 2)], dim=3)
        xx = xx.view(-1, 2*self.n_in)

        # Calculate the edge features for each node 
        # all_edges.shape = (N, K, K, n_hidden)
        all_edges = self.M(xx) 
        all_edges = all_edges.view(N, K, K, self.n_hidden)

        # Only have edges from blocks above the current block.
        # edges.shape = (N, K, n_hidden)
        mask = torch.ones(N, K, K, 1)
        if torch.cuda.is_available():
            mask = mask.cuda()
        for kx1 in range(K):
            for kx2 in range(K):
                if kx1 >= kx2:
                    mask[:, kx1, kx2, :] = 0.
        edges = torch.sum(all_edges*mask, dim=2)
        return edges

    def node_fn(self, towers, h, e):
        """
        :param towers: Node input features (N, K, n_in)
        :param h: Node input features (N, K, n_hidden)
        :param e: Node edge features (N, K, n_hidden)
        """
        N, K, _ = towers.shape

        # Concatenate all relevant inputs.
        #x = torch.cat([h, e], dim=2)
        x = torch.cat([towers, e], dim=2)
        x = x.view(-1, self.n_hidden+self.n_in)

        # Calculate the updated node features.
        x = self.U(x)
        x = x.view(N, K, self.n_hidden)
        return x

    def forward(self, towers):
        """
        :param towers: (N, K, n_in) tensor describing the tower.
        :param k: Number of times to iterate the graph update.
        """
        N, K, _ = towers.shape
        k=1
        # Initialize hidden state for each node.
        #h = self.init.expand(N, K, self.n_hidden)
        h0 = self.E(towers.view(-1, self.n_in)).view(N, K, self.n_hidden)
        h = h0
        for kx in range(k):
            # Calculate edge updates for each node: (N, K, n_hidden) 
            e = self.edge_fn(towers, h)

            # Perform node update.
            h = self.node_fn(towers, h, e)
            
        # Calculate output predictions.
        x = torch.mean(h, dim=1)
        x = self.O(x).view(N)
        return torch.sigmoid(x).unsqueeze(-1)
