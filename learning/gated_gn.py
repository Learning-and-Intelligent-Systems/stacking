import torch
from torch import nn
from torch.nn import functional as F

class GatedGN(nn.Module):
    def __init__(self, n_in, n_hidden):
        """ This network is given input of size (N, K, n_in) where N, K can vary per batch.
        :param n_in: Number of block-specific parameters.
        :param n_hidden: Number of hidden units unsed throughout the network.
        """
        super(GatedGN, self).__init__()

        self.E_n = nn.Linear(n_in, n_hidden)
        self.E_e = nn.Linear(n_hidden, n_hidden)
        self.E_g = nn.Linear(n_hidden, n_hidden)

        # Takes in features from nodes, node states, edge state, and global state.
        self.M = nn.Sequential(nn.Linear(4*n_hidden+2*n_in, n_hidden),
                               nn.ReLU(),
                               nn.Linear(n_hidden, n_hidden),
                               nn.ReLU(),
                               nn.Linear(n_hidden, n_hidden))
        self.M_update = nn.Sequential(nn.Linear(4*n_hidden+2*n_in, n_hidden),
                                      nn.Sigmoid())   
        self.M_reset = nn.Linear(4*n_hidden+2*n_in, n_hidden)

        # Update function that updates a node based on the sum of its messages.
        self.U = nn.Sequential(nn.Linear(n_in+3*n_hidden, n_hidden),            
                               nn.ReLU(),
                               nn.Linear(n_hidden, n_hidden),
                               nn.ReLU(),
                               nn.Linear(n_hidden, n_hidden))
        self.U_update = nn.Sequential(nn.Linear(n_in+3*n_hidden, n_hidden),
                                      nn.Sigmoid())
        self.U_reset = nn.Sequential(nn.Linear(n_in+3*n_hidden, n_hidden),
                                     nn.Sigmoid())

        # Recurrent function to update the global state.
        self.G = nn.Sequential(nn.Linear(3*n_hidden, n_hidden),            
                               nn.ReLU(),
                               nn.Linear(n_hidden, n_hidden),
                               nn.ReLU(),
                               nn.Linear(n_hidden, n_hidden))
        self.G_update = nn.Sequential(nn.Linear(3*n_hidden, n_hidden),
                                      nn.Sigmoid())
        self.G_reset = nn.Sequential(nn.Linear(3*n_hidden, n_hidden),
                                     nn.Sigmoid())

        # Output function that predicts stability.
        self.O = nn.Sequential(nn.Linear(n_hidden, n_hidden),
                               nn.ReLU(),
                               nn.Linear(n_hidden, n_hidden),
                               nn.ReLU(),
                               nn.Linear(n_hidden, 1))
        
        
        self.init_e = torch.zeros(1, 1, 1, n_hidden)
        self.init_g = torch.zeros(1, n_hidden)
        self.n_in, self.n_hidden = n_in, n_hidden

    def edge_fn(self, towers, h, e, g):
        N, K, _ = towers.shape

        # Get features between all node. 
        x = torch.cat([towers, h], dim=2)
        x = x[:, :, None, :].expand(N, K, K, self.n_in+self.n_hidden)
        g = g[:, None, None, :].expand(N, K, K, self.n_hidden)
        xx = torch.cat([x, x.transpose(1, 2), g, e], dim=3)
        xx = xx.view(-1, 2*self.n_in + 4*self.n_hidden)

        # Calculate gate values.
        resets = self.M_reset(xx).view(N, K, K, self.n_hidden)
        update = self.M_update(xx).view(N, K, K, self.n_hidden)

        # Apply reset gate to inputs edge state.
        xx = torch.cat([x, x.transpose(1, 2), g, resets*e], dim=3)
        xx = xx.view(-1, 2*self.n_in + 4*self.n_hidden)

        # Calculate the updated hidden node values.
        new_e = self.M(xx).view(N, K, K, self.n_hidden)
        new_e = (1 - update)*e + update*new_e
        return new_e
        
    def node_fn(self, towers, h, e, g):
        """
        :param towers: Node input features (N, K, n_in)
        :param h: Node input features (N, K, n_hidden)
        :param e: Node edge features (N, K, n_hidden)
        """
        N, K, _ = towers.shape

        # Combine the edges relevant for each node.
        mask = torch.ones(1, K, K, 1)
        if torch.cuda.is_available():
            mask = mask.cuda()
        for kx1 in range(K):
            mask[:, kx1, kx1, :] = 0.
        edges = torch.sum(e*mask, dim=2)

        # Concatenate all relevant inputs.
        g = g[:, None, :].expand(N, K, self.n_hidden)
        x = torch.cat([towers, edges, g, h], dim=2)
        x = x.view(-1, 3*self.n_hidden+self.n_in)

        # Calculate gate values.
        resets = self.U_reset(x).view(N, K, self.n_hidden)
        update = self.U_update(x).view(N, K, self.n_hidden)

        # Apply reset gate.
        x = torch.cat([towers, edges, g, resets*h], dim=2)
        x = x.view(-1, 3*self.n_hidden+self.n_in)

        # Calculate the updated node features.
        new_h = self.U(x).view(N, K, self.n_hidden)
        new_h = (1 - update)*h + update*new_h
        return new_h

    def global_fn(self, h, e, g):
        N, K, _ = h.shape

        # Concatenate all relevant inputs.
        h = h.sum(dim=1)
        e = e.sum(dim=[1,2])
        x = torch.cat([h, e, g], dim=1)

        # Calculate gate values.
        reset = self.G_reset(x)
        update = self.G_update(x)

        # Reset hidden states.
        x = torch.cat([h, e, reset*g], dim=1)

        # Calculate updated global feature.
        new_g = self.G(x)
        new_g = (1 - update)*g + update*new_g
        return new_g

    def forward(self, towers, k):
        """
        :param towers: (N, K, n_in) tensor describing the tower.
        :param k: Number of times to iterate the graph update.
        """
        N, K, _ = towers.shape
        # Initialize hidden state for each node.
        e = self.init_e.expand(N, K, K, self.n_hidden)
        g = self.init_g.expand(N, self.n_hidden)

        h = self.E_n(towers.view(-1, self.n_in)).view(N, K, self.n_hidden)
        e = self.E_e(e.view(-1, self.n_hidden)).view(N, K, K, self.n_hidden)
        g = self.E_g(g)
        
        for kx in range(k):
            # Calculate the new edge states: (N, K, K, n_hidden) 
            e = self.edge_fn(towers, h, e, g)

            # Perform node update.
            h = self.node_fn(towers, h, e, g)

            # Perform global update.
            g = self.global_fn(h, e, g)
            
        # Calculate output predictions.
        x = self.O(g).view(N)
        return torch.sigmoid(x)
        
        