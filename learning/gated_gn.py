import torch
from torch import nn
from torch.nn import functional as F




class GatedGN(nn.Module):
    def __init__(self, n_in, n_hidden, n_layers=2):
        """ This network is given input of size (N, K, n_in) where N, K can vary per batch.
        :param n_in: Number of block-specific parameters.
        :param n_hidden: Number of hidden units unsed throughout the network.
        """
        super(GatedGN, self).__init__()

        self.E_n = nn.Linear(n_in, n_hidden)
        self.E_e = nn.Linear(n_hidden, n_hidden)
        self.E_g = nn.Linear(n_hidden, n_hidden)

        # Takes in features from nodes, node states, edge state, and global state.
        self.U_gru = nn.GRU(input_size=n_in+n_hidden,
                            hidden_size=n_hidden,
                            num_layers=n_layers,
                            batch_first=True)
        self.M_gru = nn.GRU(input_size=2*n_hidden,
                            hidden_size=n_hidden,
                            num_layers=n_layers,
                            batch_first=True)
        self.G_gru = nn.GRU(input_size=2*n_hidden,
                            hidden_size=n_hidden,
                            num_layers=n_layers,
                            batch_first=True)
       
        # Output function that predicts stability.
        self.O = nn.Sequential(nn.Linear(n_hidden, n_hidden),
                               nn.ReLU(),
                               nn.Linear(n_hidden, n_hidden),
                               nn.ReLU(),
                               nn.Linear(n_hidden, 1))
        
        self.n_in, self.n_hidden, self.n_layers = n_in, n_hidden, n_layers

    def edge_fn(self, h, e_state):
        N, K, _ = h.shape

        # Get features between all node. 
        x = h[:, :, None, :].expand(N, K, K, self.n_hidden)
        xx = torch.cat([x, x.transpose(1, 2)], dim=3)
        xx = xx.view(-1, 2*self.n_hidden)

        output, e_n = self.M_gru(input=xx.view(-1, 1, 2*self.n_hidden),
                                 hx=e_state)
        output = output.view(N, K, K, self.n_hidden)
        return output, e_n

    def node_fn(self, towers, e, h_state):
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
            for kx2 in range(K):
                if kx1 != kx2 - 1:
                    mask[:, kx1, kx2, :] = 0.
        edges = torch.sum(e*mask, dim=2)

        x = torch.cat([towers, edges], dim=2)
        x = x.view(-1, self.n_in+self.n_hidden)
        
        output, h_n = self.U_gru(input=x.view(-1, 1, self.n_in+self.n_hidden),
                                 hx=h_state)
        output = output.view(N, K, self.n_hidden)
        return output, h_n

    def global_fn(self, h, e, g_state):
        N, K, _ = h.shape

        # Concatenate all relevant inputs.
        h = h.sum(dim=1)
        e = e.sum(dim=[1,2])

        x = torch.cat([h, e], dim=1).view(-1, 1, 2*self.n_hidden)
        output, g_n = self.G_gru(input=x,
                                 hx=g_state)
        output = output.view(N, self.n_hidden)
        return output, g_n

    def forward(self, towers, k):
        """
        :param towers: (N, K, n_in) tensor describing the tower.
        :param k: Number of times to iterate the graph update.
        """
        N, K, _ = towers.shape
        # Initialize hidden state for each node.
        e_state = torch.zeros(self.n_layers, N*K*K, self.n_hidden)
        h_state = torch.zeros(self.n_layers, N*K, self.n_hidden)
        g_state = torch.zeros(self.n_layers, N, self.n_hidden)
        if torch.cuda.is_available():
            e_state = e_state.cuda()
            h_state = h_state.cuda()
            g_state = g_state.cuda()
        
        h = self.E_n(towers.view(-1, self.n_in)).view(N, K, self.n_hidden)
        for kx in range(k):
            # Calculate the new edge states: (N, K, K, n_hidden) 
            e, e_state = self.edge_fn(h, e_state)

            # Perform node update.
            h, h_state = self.node_fn(towers, e, h_state)

            # Perform global update.
            g, g_state = self.global_fn(h, e, g_state)
            
        # Calculate output predictions.
        x = self.O(g).view(N)
        return torch.sigmoid(x)
        
        