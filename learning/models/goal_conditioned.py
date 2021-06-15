import torch
from torch import nn

    
class HeuristicGNN(nn.Module):
    def __init__(self, n_of_in=7, n_hidden=1):
        """ This network is given an input of size (N, K, K).
        N is the batch size, K is the number of objects (including a * object).
        The model is one interation of message passing in a GNN
        :param n_of_in: Dimensionality of object features (one-hot encodings)
        :param n_hidden: Number of hidden units used throughout the network.
        """
        super(HeuristicGNN, self).__init__()

        torch.set_default_dtype(torch.float64) # my data was float64 and model params were float32
        
        # Message function that compute relation between two nodes and outputs a message vector.
        self.E = nn.Sequential(nn.Linear(2*n_of_in, n_hidden),
                               nn.Tanh())

        # Update function that updates a node based on the sum of its messages.
        self.N = nn.Sequential(nn.Linear(n_of_in+n_hidden, n_hidden),  
                               nn.Tanh())

        # Output function that predicts heuristic.
        self.O = nn.Sequential(nn.Linear(2*n_hidden, n_hidden),
                               nn.Tanh(),
                               nn.Linear(n_hidden, 1))

        self.n_of_in, self.n_hidden = n_of_in, n_hidden
        
    def node_fn(self, object_features, e):
        """
        :param object_features: Node input features (N, K, n_of_in)
        :param e: Node edge features (N, K, K, n_hidden)
        """
        N, K, n_of_in = object_features.shape

        # sum all edge features going into each node
        he = torch.sum(e, dim=2)
        # Concatenate all relevant inputs.
        x = torch.cat([object_features, he], dim=2)
        x = x.view(-1, self.n_hidden+self.n_of_in)

        # Calculate the updated node features. 
        x = self.N(x)
        x = x.view(N, K, self.n_hidden)
        return x

    def edge_fn(self, object_features, edge_mask):
        """
        :param edge_mask: Mask on edges between objects (N, K, K)
        :param hn: Node hidden states
        """
        N, K, n_of_in = object_features.shape
        N, K, K = edge_mask.shape
        edge_mask = edge_mask[:, :, :, None].expand(N, K, K, 1)

        # Get features between all node. 
        # object_features.shape = (N, K, n_of_in)
        # x.shape = (N, K, K, n_of_in)
        # xx.shape = (N, K, K, 2*n_of_in) --> (N*K*K, 2*n_of_in)
        x = object_features[:, :, None, :].expand(N, K, K, self.n_of_in)
        xx = torch.cat([x, x.transpose(1, 2)], dim=3)
        xx = xx.view(-1, 2*self.n_of_in)
        
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

    def encode_state(self, object_features, edge_mask):
        """
        :param edge_mask: state edge mask (N, K, K)
        """
        N, K, K = edge_mask.shape

        # Calculate edge updates for each node with action as input
        he = self.edge_fn(object_features, edge_mask)
        
        # Perform node update.
        hn = self.node_fn(object_features, he)
        
        # Calculate output predictions.
        x = torch.mean(hn, dim=1)
        return x
        
    def forward(self, x):
        state_edge_mask, goal_edge_mask = x
        N, K, K = state_edge_mask.shape
        
        # TODO: don't hard code all object properties
        # object_features.shape = (N, K, n_of_in)
        # object_features = torch.arange(K, dtype=torch.float64, requires_grad=True)
        # object_features = object_features.unsqueeze(dim=1)
        object_features = torch.eye(K, dtype=torch.float64, requires_grad=True)
        object_features = object_features.repeat(N, 1, 1)
        
        h_state = self.encode_state(object_features, state_edge_mask)
        h_goal = self.encode_state(object_features, goal_edge_mask)
        
        pred = self.O(torch.cat([h_state, h_goal], dim=1)).view(N)
        return pred


class TransitionGNN(nn.Module):
    def __init__(self, args, n_ef_in=7, n_af_in=1, n_hidden=16):
        """ This network is given three inputs of size (N, K, K), (N, 1), and (N, K, K).
        N is the batch size, K is the number of objects (including a table)
        :param n_ef_in: Dimensionality of edge features
        :param n_af_in: Dimensionality of action features
        :param n_hidden: Number of hidden units used throughout the network.
        """
        super(TransitionGNN, self).__init__()

        torch.set_default_dtype(torch.float64) # my data was float64 and model params were float32
        
        # Initial embedding of edge features and action into latent state
        self.Ei = nn.Sequential(nn.Linear(n_ef_in+n_af_in, n_hidden))#,
                               #nn.Tanh())
        
        # Update function that updates a node based on the sum of its messages.
        self.N = nn.Sequential(nn.Linear(n_hidden, n_hidden),  
                               nn.Tanh())#,
                               #nn.Linear(n_hidden, n_hidden),
                               #nn.Tanh())
            
        # Message function that compute relation between two nodes and outputs a message vector.
        self.E = nn.Sequential(nn.Linear(2*n_hidden, n_hidden),
                               nn.Tanh())#,
                               #nn.Linear(n_hidden, n_hidden),
                               #nn.Tanh())
                   
        # Final function to get next state edge predictions
        self.Ef = nn.Sequential(nn.Linear(n_hidden, n_hidden),
                                nn.ReLU(),
                                nn.Linear(n_hidden, n_hidden),
                                nn.ReLU(),
                                nn.Linear(n_hidden,n_ef_in),
                                nn.Tanh())

        self.n_ef_in, self.n_af_in, self.n_hidden = n_ef_in, n_af_in, n_hidden
        
        self.pred_type = args.pred_type

    def embed(self, edge_features, action):
        """
        :param edge_features: Node input features (N, K, K, n_ef_in)
        :param action: Action taken (N, n_af_in)
        """
        N, K, K, n_ef_in = edge_features.shape
        N, n_af_in = action.shape
        
        # Append action to edge features
        # a.shape = (N, K, K, n_af_in)
        # xa.shape = (N, K, K, n_ef_in+n_af_in) --> (N*K*K, n_ef_in+n_af_in)
        a = action[:, None, None, :].expand(-1, K, K, -1)
        xa = torch.cat([edge_features, a], dim=3)
        xa = xa.view(-1, n_ef_in+n_af_in)
        
        # Calculate the hidden edge state for each node
        # he.shape = (N*K*K, n_hidden) --> (N, K, K, n_hidden)
        he = self.Ei(xa).view(N, K, K, self.n_hidden)
        return he

    def node_fn(self, he):
        """
        :param he: Hidden edge features (N, K, K, n_hidden)
        """
        N, K, K, n_hidden = he.shape

        # sum all edge features going into each node
        # he_sum.shape (N, K, n_hidden) --> (N*K, n_hidden)
        he_sum = torch.sum(he, dim=2).view(-1, n_hidden)

        # Calculate the updated node features.
        # hn.shape = (N, K, n_hidden)
        hn = self.N(he_sum).view(N, K, n_hidden)
        return hn

    def edge_fn(self, hn):
        """
        :param hn: Node hidden states (N, K, n_hidden)
        """
        N, K, n_hidden = hn.shape
        
        # Get features between all nodes and mask with edge features.
        # x.shape = (N, K, K, n_hidden)
        # xx.shape = (N, K, K, 2*n_hidden) --> (N*K*K, 2*n_hidden)
        x = hn[:, :, None, :].expand(-1, -1, K, -1)
        xx = torch.cat([x, x.transpose(1, 2)], dim=3)
        
        # Calculate the hidden edge state for each edge
        # he.shape = (N, K, K, n_hidden)
        he = self.E(xx).view(N, K, K, self.n_hidden)
        return he

    def forward(self, x):
        """
        :param x: list of object features (N, K, n_of_in),
                            edge features (N, K, K, n_ef_in),
                            action (N, n_of_in)
        """
        
        edge_features, action = x
        N, K, K, n_ef_in = edge_features.shape
        N, n_af_in = action.shape

        # Calculate initial node hidden state
        he = self.embed(edge_features, action)

        I = 1
        for i in range(I):
            # Calculate node hidden state
            hn = self.node_fn(he)
            
            # Calculate edge hidden state
            he = self.edge_fn(hn)

        # Calculate the final edge predictions
        # he.shape = (N, K, K, n_hidden) --> (N*K*K, n_hidden)
        he = he.view(-1, self.n_hidden)
        y = self.Ef(he).view(N, K, K, n_ef_in)
        
        # if predicting next full state, hidden state is a probability
        if self.pred_type == 'full_state':
            y = torch.sigmoid(y)
        
        return y