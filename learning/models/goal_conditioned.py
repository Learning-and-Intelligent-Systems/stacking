import torch
from torch import nn

        
class HeuristicGNN(nn.Module):
    def __init__(self, n_in=7, n_hidden=1):
        """ This network is given an input of size (N, K, K).
        N is the batch size, K is the number of objects (including a * object).
        The model is one interation of message passing in a GNN
        :param n_hidden: Number of hidden units used throughout the network.
        """
        super(HeuristicGNN, self).__init__()

        torch.set_default_dtype(torch.float64) # my data was float64 and model params were float32
        
        # Message function that compute relation between two nodes and outputs a message vector.
        self.E = nn.Sequential(nn.Linear(2*n_in, n_hidden),
                               nn.ReLU())

        # Update function that updates a node based on the sum of its messages.
        self.N = nn.Sequential(nn.Linear(n_in+n_hidden, n_hidden),  
                               nn.ReLU())

        # Output function that predicts heuristic.
        self.O = nn.Sequential(nn.Linear(2*n_hidden, n_hidden),
                               nn.ReLU(),
                               nn.Linear(n_hidden, 1))

        self.n_in, self.n_hidden = n_in, n_hidden
        
    def node_fn(self, object_states, e):
        """
        :param object_states: Node input features (N, K, n_in)
        :param e: Node edge features (N, K, K, n_hidden)
        """
        N, K, n_in = object_states.shape

        # sum all edge features going into each node
        he = torch.sum(e, dim=2)
        # Concatenate all relevant inputs.
        x = torch.cat([object_states, he], dim=2)
        x = x.view(-1, self.n_hidden+self.n_in)

        # Calculate the updated node features. 
        x = self.N(x)
        x = x.view(N, K, self.n_hidden)
        return x

    def edge_fn(self, object_states, edge_mask):
        """
        :param edge_mask: Mask on edges between objects (N, K, K)
        :param hn: Node hidden states
        """
        N, K, n_in = object_states.shape
        N, K, K = edge_mask.shape
        edge_mask = edge_mask[:, :, :, None].expand(N, K, K, 1)

        # Get features between all node. 
        # object_states.shape = (N, K, n_in)
        # x.shape = (N, K, K, n_in)
        # xx.shape = (N, K, K, 2*n_in) --> (N*K*K, 2*n_in)
        x = object_states[:, :, None, :].expand(N, K, K, self.n_in)
        xx = torch.cat([x, x.transpose(1, 2)], dim=3)
        xx = xx.view(-1, 2*self.n_in)
        
        # Calculate the edge features
        # all_edges.shape = (N, K, K, 1)
        all_edges = self.E(xx) 
        all_edges = all_edges.view(N, K, K, 1)

        # Use edge mask to calculate edge features and sum features for each node
        # edges.shape = (N, K, K, 1)
        if torch.cuda.is_available():
            edge_mask = edge_mask.cuda()
        all_edges = all_edges*edge_mask
        return all_edges

    def encode_state(self, object_states, edge_mask):
        """
        :param edge_mask: state edge mask (N, K, K)
        """
        N, K, K = edge_mask.shape

        # Calculate edge updates for each node with action as input
        he = self.edge_fn(object_states, edge_mask)
        
        # Perform node update.
        hn = self.node_fn(object_states, he)
        
        # Calculate output predictions.
        x = torch.mean(hn, dim=1)
        return x
        
    def forward(self, x):
        state_edge_mask, goal_edge_mask = x
        N, K, K = state_edge_mask.shape
        
        # TODO: don't hard code all object properties
        # object_states.shape = (N, K, n_in)
        # object_states = torch.arange(K, dtype=torch.float64, requires_grad=True)
        # object_states = object_states.unsqueeze(dim=1)
        object_states = torch.eye(K, dtype=torch.float64, requires_grad=True)
        object_states = object_states.repeat(N, 1, 1)
        
        h_state = self.encode_state(object_states, state_edge_mask)
        h_goal = self.encode_state(object_states, goal_edge_mask)
        
        pred = self.O(torch.cat([h_state, h_goal], dim=1)).view(N)
        return pred

class TransitionGNN(nn.Module):
    def __init__(self, args, n_in=7, n_hidden=1):
        """ This network is given three inputs of size (N, K, K), (N, 1), and (N, K, K).
        N is the batch size, K is the number of objects (including a * object)
        :param n_in: Dimensionality of object state and action (one-hot encodings)
        :param n_hidden: Number of hidden units used throughout the network.
        """
        super(TransitionGNN, self).__init__()

        torch.set_default_dtype(torch.float64) # my data was float64 and model params were float32
        
        # Message function that compute relation between two nodes and outputs a message vector.
        self.E1 = nn.Sequential(nn.Linear(3*n_in, n_hidden),
                               nn.ReLU())
        self.E2 = nn.Sequential(nn.Linear(2*n_hidden, 1),
                               nn.ReLU())

        # Update function that updates a node based on the sum of its messages.
        self.N = nn.Sequential(nn.Linear(n_in+n_hidden, n_hidden),  
                               nn.ReLU())

        self.n_in, self.n_hidden = n_in, n_hidden
        
        self.pred_type = args.pred_type

    def action_edge_fn(self, object_states, edge_mask, action):
        """
        :param object_states: Node input features (K, n_in)
        :param edge_mask: Mask on edges between objects (N, K, K)
        :param action: Action taken (N)
        """
        N, K, K = edge_mask.shape
        edge_mask = edge_mask[:, :, :, None].expand(N, K, K, 1)

        # Get features between all node. 
        # action.shape = (N, n_in)
        # object_states.shape = (N, K, n_in)
        # a.shape = (N, K, K, n_in)
        # x.shape = (N, K, K, n_in)
        # xx.shape = (N, K, K, 2*n_in)
        # xxa.shape = (N, K, K, 3*n_in) --> (N*K*K, 3*n_in)
        a = action[:, None, None, :].expand(-1, K, K, -1)
        x = object_states[:, :, None, :].expand(N, K, K, self.n_in)
        xx = torch.cat([x, x.transpose(1, 2)], dim=3)
        xxa = torch.cat([xx, a], dim=3)
        xxa = xxa.view(-1, 3*self.n_in)
        
        # Calculate the edge features for each node 
        # all_edges.shape = (N, K, K, n_hidden)
        all_edges = self.E1(xxa) 
        all_edges = all_edges.view(N, K, K, self.n_hidden)

        # Use edge mask to calculate edge features and sum features for each node
        # edges.shape = (N, K, n_hidden)
        if torch.cuda.is_available():
            edge_mask = edge_mask.cuda()
        edges = all_edges*edge_mask
        return edges

    def node_fn(self, object_states, e):
        """
        :param object_states: Node input features (N, K, n_in)
        :param e: Node edge features (N, K, n_hidden)
        """
        N, K, n_in = object_states.shape

        # sum all edge features going into each node
        he = torch.sum(e, dim=2)
        # Concatenate all relevant inputs.
        x = torch.cat([object_states, he], dim=2)
        x = x.view(-1, self.n_hidden+self.n_in)

        # Calculate the updated node features. 
        x = self.N(x)
        x = x.view(N, K, self.n_hidden)
        return x

    def final_edge_fn(self, edge_mask, hn):
        """
        :param edge_mask: Mask on edges between objects (N, K, K)
        :param hn: Node hidden states
        """
        N, K, K = edge_mask.shape
        edge_mask = edge_mask[:, :, :, None].expand(N, K, K, 1)

        # Get features between all node. 
        # hn.shape = (N, K, n_hidden)
        # x.shape = (N, K, K, n_hidden)
        # xx.shape = (N, K, K, 2*n_hidden) --> (N*K*K, 2*n_hidden)
        x = hn[:, :, None, :].expand(N, K, K, self.n_hidden)
        xx = torch.cat([x, x.transpose(1, 2)], dim=3)
        xx = xx.view(-1, 2*self.n_hidden)
        
        # Calculate the edge features
        # all_edges.shape = (N, K, K, 1)
        all_edges = self.E2(xx) 
        all_edges = all_edges.view(N, K, K, 1)

        # Use edge mask to calculate edge features and sum features for each node
        # edges.shape = (N, K, K, 1)
        if torch.cuda.is_available():
            edge_mask = edge_mask.cuda()
        #all_edges = all_edges*edge_mask
        
        if self.pred_type == 'full_state':
            all_edges = torch.sigmoid(all_edges)
        all_edges = all_edges.squeeze(-1)
        return all_edges

    def forward(self, x):
        """
        :param x: list of state edge mask (N, K, K) and action (N)
        """
        
        edge_mask, action = x
        N = action.shape
        
        N, K, K = edge_mask.shape
        
        # TODO: don't hard code all object properties
        # object_states.shape = (N, K, n_in)
        # object_states = torch.arange(K, dtype=torch.float64, requires_grad=True)
        # object_states = object_states.unsqueeze(dim=1)
        object_states = torch.eye(K, dtype=torch.float64, requires_grad=True)
        object_states = object_states.repeat(N, 1, 1)

        # Calculate edge updates for each node with action as input
        he1 = self.action_edge_fn(object_states, edge_mask, action)
        
        # Perform node update.
        hn = self.node_fn(object_states, he1)
        
        # Calculate final edge features
        he2 = self.final_edge_fn(edge_mask, hn)
        return he2