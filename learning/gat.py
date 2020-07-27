""" Massachusetts Institute of Technology

Izzy Brand, 2020
"""
import torch
from torch import nn
from torch.nn import functional as F

class FCGAT(nn.Module):
    """ Implements graph attention networks as introduced in
    https://arxiv.org/abs/1710.10903

    Takes in a set of nodes from a fully connected graph. Applies a
    graph attention network layer. The set of nodes is passed in as a 
    [N x K x D] tensor. N batches, K nodes, D dimension at each node.

    The graph is assumed to be fully connected, so no edge mask is applied.
    This implementation uses single-headed attention. Furthermore, we assume
    that the output feature dimension is equal to the input dimension.
    
    Extends:
        nn.Module
    """

    def __init__(self, D):
        """
        Arguments:
            D {int} -- dimension of node features
        """
        super(FCGAT, self).__init__()

        self.W = nn.Linear(D, D)

        self.fc_attention = nn.Sequential(
            nn.Linear(2*D, 1),
            nn.LeakyReLU()
        )

    def attention(self, x):
        """ Self attention layer. Outputs attention between pairs of nodes

        Arguments:
            x {torch.Tensor} -- [N x K x D] tensor of nodes
        
        Returns:
            torch.Tensor -- [N x K x K x A] tensor of attention weights
        """
        N, K, D = x.shape
        # create an [N x K x K x 2D] vector of the pairs of node features
        x = x[:, :, None, :].expand(N, K, K, D)
        xx = torch.stack([x, x.transpose(1,2)], dim=3)
        # flatten, apply attention weights, and drop the extra dimension
        aa = self.fc_attention(xx.view(-1, 2*D))[..., 0]
        # unflatten and normalize attention weights for each node
        return F.softmax(aa.view(N, K, K), dim=2)
        
    def forward(self, x):

        N, K, D = x.shape
        # apply the weight matrix to the node features
        x = self.W(x.view(-1, D)).view(N, K, D)
        # get an attention mask for each node
        a = self.attention(x)
        # apply the attention mask to the nodes features
        x = torch.einsum('nkj, nkd -> nkd', a, x)
        # and apply a nonlinearity
        return F.relu(x)

