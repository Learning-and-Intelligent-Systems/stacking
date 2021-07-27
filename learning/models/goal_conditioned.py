import torch
from torch import nn


class GCGNN(nn.Module):
    def __init__(self, n_of_in=7, n_ef_in=7, n_hidden=16):
        """
        :param n_of_in: Dimensionality of object features
        :param n_ef_in: Dimensionality of edge features
        :param n_hidden: Number of hidden units used throughout the network.
        """
        super().__init__()

        torch.set_default_dtype(torch.float64) # my data was float64 and model params were float32

        self.n_of_in, self.n_ef_in, self.n_hidden = n_of_in, n_ef_in, n_hidden

        # Initial embedding of node features into latent state
        self.Ni = nn.Sequential(nn.Linear(n_of_in, n_hidden))#,
                               #nn.Tanh())

        # Message function that compute relation between two nodes and outputs a message vector.
        self.E = nn.Sequential(nn.Linear(2*n_hidden, n_hidden),
                               nn.Tanh())#,
                               #nn.Linear(n_hidden, n_hidden),
                               #nn.Tanh())

        # Update function that updates a node based on the sum of its messages.
        self.N = nn.Sequential(nn.Linear(n_hidden, n_hidden),
                               nn.Tanh())#,
                               #nn.Linear(n_hidden, n_hidden),
                               #nn.Tanh())

    def embed_node(self, input):
        """
        :param input: Network input. First item is always
                        object_features (N, K, n_of_in)
        """
        object_features = input[0]
        N, K, n_of_in = object_features.shape

        # Pass each object feature from encoder
        # x.shape = (N*K, n_of_in)
        x = object_features.view(-1, n_of_in)

        # Calculate the hidden state for each node
        # hn.shape = (N*K, n_hidden) --> (N, K, n_hidden)
        hn = self.Ni(x).view(N, K, self.n_hidden)
        return hn

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

    def forward(self, input):
        """
        :param input: network inputs. First element in list is always object_features
        """

        # Calculate initial node and edge hidden states
        hn = self.embed_node(input)
        he = self.embed_edge(hn, input)

        I = 1
        for i in range(I):
            # Calculate node hidden state
            hn = self.node_fn(he)

            # Calculate edge hidden state
            he = self.edge_fn(hn)

        y = self.final_pred(he)
        return y

class HeuristicGNN(GCGNN):
    def __init__(self, n_of_in=7, n_ef_in=7, n_hidden=16):
        """ This network is given three inputs of size (N, K, n_of_in), (N, K, K, n_ef_in), and (N, n_af_in).
        N is the batch size, K is the number of objects (including a table)
        :param n_of_in: Dimensionality of object features
        :param n_ef_in: Dimensionality of edge features
        :param n_hidden: Number of hidden units used throughout the network.
        """
        super(HeuristicGNN, self).__init__(n_of_in, n_ef_in, n_hidden)

        # Initial embedding of edge features and action into latent state
        self.Ei = nn.Sequential(nn.Linear(2*n_ef_in+2*n_hidden, n_hidden))#,
                               #nn.Tanh())

        # Final function to get next state edge predictions
        self.Ef = nn.Sequential(nn.Linear(n_hidden, n_hidden),
                                nn.ReLU(),
                                nn.Linear(n_hidden, n_hidden),
                                nn.ReLU(),
                                nn.Linear(n_hidden, 1),
                                nn.Tanh())

    def embed_edge(self, hn, input):
        """
        :param hn: Hidden node state (N, K, n_hidden)
        :param input: Network inputs. Elements are
            object_features (N, K, n_hidden)
            edge_features (N, K, K, n_ef_in)
            goal_edge_features (N, K, K, n_ef_in)
        """
        object_features, edge_features, goal_edge_features = input
        N, K, n_hidden = hn.shape
        N, K, n_of_in = object_features.shape
        N, K, K, n_ef_in = edge_features.shape
        N, K, K, n_ef_in = goal_edge_features.shape

        # Append edge features, goal edge features, and hidden node states
        # a.shape = (N, K, K, n_af_in)
        # hn_exp.shape = (N, K, K, n_hidden)
        # hnhn.shape = (N, K, K, 2*n_hidden)
        # xahnhn.shape = (N, K, K, n_ef_in+n_af_in+2*n_hidden) --> (N*K*K, n_ef_in+n_af_in+2*n_hidden)
        # gxhnhn.shape = (N, K, K, 2*n_ef_in+2*n_hidden) --> (N*K*K, 2*n_ef_in+2*n_hidden)
        hn_exp = hn[:, :, None, :].expand(-1, -1, K, -1)
        hnhn = torch.cat([hn_exp, hn_exp.transpose(1, 2)], dim=3)
        gxhnhn = torch.cat([edge_features, goal_edge_features, hnhn], dim=3)
        gxhnhn = gxhnhn.view(-1, 2*n_ef_in+2*n_hidden)

        # Calculate the hidden edge state for each node
        # he.shape = (N*K*K, n_hidden) --> (N, K, K, n_hidden)
        he = self.Ei(gxhnhn).view(N, K, K, self.n_hidden)
        return he

    def final_pred(self, he):
        N, K, K, n_hidden = he.shape

        # Calculate the final edge predictions
        # he.shape = (N, K, K, n_hidden) --> (N*K*K, n_hidden)
        he = he.view(-1, self.n_hidden)
        y = self.Ef(he).view(N, K*K)

        # sum all edge outputs to get predicted steps to goal
        y = torch.sum(y, axis=1)
        return y


class TransitionGNN(GCGNN):
    def __init__(self, n_of_in=7, n_ef_in=7, n_af_in=1, n_hidden=16, pred_type='delta_state'):
        """ This network is given three inputs of size (N, K, n_of_in), (N, K, K, n_ef_in), and (N, n_af_in).
        N is the batch size, K is the number of objects (including a table)
        :param n_of_in: Dimensionality of object features
        :param n_ef_in: Dimensionality of edge features
        :param n_af_in: Dimensionality of action features
        :param n_hidden: Number of hidden units used throughout the network.
        :param pred_type: delta_state -- predict change in edge features
                          full_state -- predict full next edge features
                          class -- predict wether or not the optimistic function is correct
        """
        super(TransitionGNN, self).__init__(n_of_in, n_ef_in, n_hidden)

        # Initial embedding of edge features and action into latent state
        self.Ei = nn.Sequential(nn.Linear(n_ef_in+n_af_in+2*n_hidden, n_hidden))#,
                               #nn.Tanh())

        # Final function to get next state edge predictions
        self.Ef = nn.Sequential(nn.Linear(n_hidden, n_hidden),
                                nn.ReLU(),
                                nn.Linear(n_hidden, n_hidden),
                                nn.ReLU(),
                                nn.Linear(n_hidden,n_ef_in),
                                nn.Tanh())

        self.n_af_in = n_af_in
        self.pred_type = pred_type

    def embed_edge(self, hn, input):
        """
        :param hn: Hidden node state (N, K, n_hidden)
        :param input: Network inputs. Elements are
            object_features (N, K, n_hidden)
            edge_features (N, K, K, n_ef_in)
            action (N, n_af_in)
        """
        object_features, edge_features, action = input
        N, K, n_hidden = hn.shape
        N, K, n_of_in = object_features.shape
        N, K, K, n_ef_in = edge_features.shape
        N, n_af_in = action.shape

        # Append edge features, action, and hidden node states
        # a.shape = (N, K, K, n_af_in)
        # hn_exp.shape = (N, K, K, n_hidden)
        # hnhn.shape = (N, K, K, 2*n_hidden)
        # xahnhn.shape = (N, K, K, n_ef_in+n_af_in+2*n_hidden) --> (N*K*K, n_ef_in+n_af_in+2*n_hidden)
        a = action[:, None, None, :].expand(-1, K, K, -1)
        hn_exp = hn[:, :, None, :].expand(-1, -1, K, -1)
        hnhn = torch.cat([hn_exp, hn_exp.transpose(1, 2)], dim=3)
        xahnhn = torch.cat([edge_features, a, hnhn], dim=3)
        xahnhn = xahnhn.view(-1, n_ef_in+n_af_in+2*n_hidden)

        # Calculate the hidden edge state for each node
        # he.shape = (N*K*K, n_hidden) --> (N, K, K, n_hidden)
        he = self.Ei(xahnhn).view(N, K, K, self.n_hidden)
        return he

    def final_pred(self, he):
        N, K, K, n_hidden = he.shape

        if self.pred_type == 'delta_state' or self.pred_type == 'full_state':
            # Calculate the final edge predictions
            # he.shape = (N, K, K, n_hidden) --> (N*K*K, n_hidden)
            he = he.view(-1, self.n_hidden)
            y = self.Ef(he).view(N, K, K, self.n_ef_in)

            # if predicting next full state, hidden state is a probability
            if self.pred_type == 'full_state':
                y = torch.sigmoid(y)
        elif self.pred_type == 'class':
            # Calculate the final edge predictions
            # he.shape = (N, K, K, n_hidden)
            # x.shape = (N, n_hidden)
            # y.shape = (N, 1) --> (N)
            x = torch.mean(he, dim=(1,2))
            y = self.Ef(x).view(N)
            return torch.sigmoid(y)
        return y
