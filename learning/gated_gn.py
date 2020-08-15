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
        self.M_nodes = nn.Sequential(nn.Linear(2*n_hidden, n_hidden),
                                     nn.ReLU(),
                                     nn.Linear(n_hidden, n_hidden),
                                     nn.ReLU(),
                                     nn.Linear(n_hidden, n_hidden))
        self.M_edges = nn.Sequential(nn.Linear(n_hidden, n_hidden),
                                     nn.ReLU(),
                                     nn.Linear(n_hidden, n_hidden),
                                     nn.ReLU(),
                                     nn.Linear(n_hidden, n_hidden))
        self.M_global = nn.Sequential(nn.Linear(n_hidden, n_hidden),
                                      nn.ReLU(),
                                      nn.Linear(n_hidden, n_hidden),
                                      nn.ReLU(),
                                      nn.Linear(n_hidden, n_hidden))

        self.M_update_nodes = nn.Linear(2*n_hidden, n_hidden)
        self.M_update_edges = nn.Linear(n_hidden, n_hidden)
        self.M_update_global = nn.Linear(n_hidden, n_hidden)
        self.M_reset_nodes = nn.Linear(2*n_hidden, n_hidden)
        self.M_reset_edges = nn.Linear(n_hidden, n_hidden)
        self.M_reset_global = nn.Linear(n_hidden, n_hidden)

        # Update function that updates a node based on the sum of its messages.
        self.U_feats = nn.Sequential(nn.Linear(n_in, n_hidden),            
                               nn.ReLU(),
                               nn.Linear(n_hidden, n_hidden),
                               nn.ReLU(),
                               nn.Linear(n_hidden, n_hidden))
        self.U_node = nn.Sequential(nn.Linear(n_hidden, n_hidden),            
                               nn.ReLU(),
                               nn.Linear(n_hidden, n_hidden),
                               nn.ReLU(),
                               nn.Linear(n_hidden, n_hidden))
        self.U_edges = nn.Sequential(nn.Linear(n_hidden, n_hidden),            
                               nn.ReLU(),
                               nn.Linear(n_hidden, n_hidden),
                               nn.ReLU(),
                               nn.Linear(n_hidden, n_hidden))
        self.U_global = nn.Sequential(nn.Linear(n_hidden, n_hidden),            
                               nn.ReLU(),
                               nn.Linear(n_hidden, n_hidden),
                               nn.ReLU(),
                               nn.Linear(n_hidden, n_hidden))

        self.U_update_feats = nn.Linear(n_in, n_hidden)
        self.U_update_node = nn.Linear(n_hidden, n_hidden)
        self.U_update_edges = nn.Linear(n_hidden, n_hidden)
        self.U_update_global = nn.Linear(n_hidden, n_hidden)

        self.U_reset_feats = nn.Linear(n_in, n_hidden)
        self.U_reset_node = nn.Linear(n_hidden, n_hidden)
        self.U_reset_edges = nn.Linear(n_hidden, n_hidden)
        self.U_reset_global = nn.Linear(n_hidden, n_hidden)

        # Recurrent function to update the global state.
        self.G_nodes = nn.Sequential(nn.Linear(n_hidden, n_hidden),            
                               nn.ReLU(),
                               nn.Linear(n_hidden, n_hidden),
                               nn.ReLU(),
                               nn.Linear(n_hidden, n_hidden))
        self.G_edges = nn.Sequential(nn.Linear(n_hidden, n_hidden),            
                               nn.ReLU(),
                               nn.Linear(n_hidden, n_hidden),
                               nn.ReLU(),
                               nn.Linear(n_hidden, n_hidden))
        self.G_global = nn.Sequential(nn.Linear(n_hidden, n_hidden),            
                               nn.ReLU(),
                               nn.Linear(n_hidden, n_hidden),
                               nn.ReLU(),
                               nn.Linear(n_hidden, n_hidden))

        self.G_update_nodes = nn.Linear(n_hidden, n_hidden)
        self.G_update_edges = nn.Linear(n_hidden, n_hidden)
        self.G_update_global = nn.Linear(n_hidden, n_hidden)

        self.G_reset_nodes = nn.Linear(n_hidden, n_hidden)
        self.G_reset_edges = nn.Linear(n_hidden, n_hidden)
        self.G_reset_global = nn.Linear(n_hidden, n_hidden)

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
        x = h[:, :, None, :].expand(N, K, K, self.n_hidden)
        g = g[:, None, None, :].expand(N, K, K, self.n_hidden)
        xx = torch.cat([x, x.transpose(1, 2)], dim=3)
        xx = xx.view(-1, 2*self.n_hidden)

        # Calculate gate values.
        r_g = self.M_reset_global(g.reshape(-1, self.n_hidden)).reshape(N, K, K, self.n_hidden)
        r_e = self.M_reset_edges(e.view(-1, self.n_hidden)).view(N, K, K, self.n_hidden)
        r_n = self.M_reset_nodes(xx).view(N, K, K, self.n_hidden)
        reset = torch.sigmoid(r_g + r_e + r_n)

        u_g = self.M_update_global(g.reshape(-1, self.n_hidden)).reshape(N, K, K, self.n_hidden)
        u_e = self.M_update_edges(e.view(-1, self.n_hidden)).view(N, K, K, self.n_hidden)
        u_n = self.M_update_nodes(xx).view(N, K, K, self.n_hidden)
        update = torch.sigmoid(u_g + u_e + u_n)

        # Apply reset gate to inputs edge state.
        new_e_g = self.M_global(g.reshape(-1, self.n_hidden)).reshape(N, K, K, self.n_hidden)
        new_e_h = self.M_nodes(xx).view(N, K, K, self.n_hidden)
        new_e_e = self.M_edges((e*reset).view(-1, self.n_hidden)).view(N, K, K, self.n_hidden)

        # Calculate the updated hidden node values.
        new_e = torch.tanh(new_e_g + new_e_h + new_e_h)
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
        r_g = self.U_reset_global(g.reshape(-1, self.n_hidden)).reshape(N, K, self.n_hidden)
        r_e = self.U_reset_edges(edges.view(-1, self.n_hidden)).view(N, K, self.n_hidden)
        r_n = self.U_reset_node(h.view(-1, self.n_hidden)).view(N, K, self.n_hidden)
        r_f = self.U_reset_feats(towers.view(-1, self.n_in)).view(N, K, self.n_hidden)
        reset = torch.sigmoid(r_g + r_e + r_n + r_f)

        u_g = self.U_update_global(g.reshape(-1, self.n_hidden)).reshape(N, K, self.n_hidden)
        u_e = self.U_update_edges(edges.view(-1, self.n_hidden)).view(N, K, self.n_hidden)
        u_n = self.U_update_node(h.view(-1, self.n_hidden)).view(N, K, self.n_hidden)
        u_f = self.U_update_feats(towers.view(-1, self.n_in)).view(N, K, self.n_hidden)
        update = torch.sigmoid(u_g + u_e + u_n + u_f)
        
        # Calculate the updated node features.
        new_h_f = self.U_feats(towers.view(-1, self.n_in)).view(N, K, self.n_hidden)
        new_h_n = self.U_node((reset*h).view(-1, self.n_hidden)).view(N, K, self.n_hidden)
        new_h_e = self.U_edges(edges.view(-1, self.n_hidden)).view(N, K, self.n_hidden)
        new_h_g = self.U_global(g.reshape(-1, self.n_hidden)).reshape(N, K, self.n_hidden)
        new_h = torch.tanh(new_h_f + new_h_n + new_h_e + new_h_g)
        new_h = (1 - update)*h + update*new_h
        return new_h

    def global_fn(self, h, e, g):
        N, K, _ = h.shape

        # Concatenate all relevant inputs.
        h = h.sum(dim=1)
        e = e.sum(dim=[1,2])

        # Calculate gate values.
        r_g = self.G_reset_global(g)
        r_n = self.G_reset_nodes(h)
        r_e = self.G_reset_edges(e)
        reset = torch.sigmoid(r_g + r_n + r_e)

        u_g = self.G_update_global(g)
        u_n = self.G_update_nodes(g)
        u_e = self.G_update_edges(g)
        update = torch.sigmoid(u_g + u_n + u_e)

        # Calculate updated global feature.
        new_g_g = self.G_global(reset*g)
        new_g_e = self.G_edges(e)
        new_g_v = self.G_nodes(h)
        new_g = torch.tanh(new_g_g + new_g_e + new_g_v)
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
        
        