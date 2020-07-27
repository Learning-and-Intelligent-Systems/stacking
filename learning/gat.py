""" Massachusetts Institute of Technology

Izzy Brand, 2020
"""
import torch
from torch import nn
from torch.nn import functional as F

class FCGAT(nn.Module):
    """ Implements graph attention networks as introduced in
    https://arxiv.org/abs/1710.10903

    This implementation makes the following structural assumptions:
     * The graph is assumed to be fully connected, so no edge mask is applied
     * Single-headed attention

    Extends:
        nn.Module
    """

    def __init__(self, D1, D2):
        """
        Arguments:
            D1 {int} -- input dimension of node features
            D2 {int} -- output dimension of node features
        """
        super(FCGAT, self).__init__()
        self.D1 = D1
        self.D2 = D2

        # the node feature update weights
        self.W = nn.Linear(D1, D2)
        # attention weights
        self.fc_attention = nn.Sequential(
            nn.Linear(2*D2, 1),
            nn.LeakyReLU()
        )

    def attention(self, x):
        """ Self attention layer. Outputs attention between pairs of nodes

        Arguments:
            x {torch.Tensor} -- [N x K x D2] tensor of nodes

        Returns:
            torch.Tensor -- [N x K x K] tensor of attention weights
        """
        N, K, _ = x.shape
        # create an [N x K x K x 2D2] vector of the pairs of node features
        x = x[:, :, None, :].expand(N, K, K, self.D2)
        xx = torch.stack([x, x.transpose(1,2)], dim=3)
        # flatten, apply attention weights, and drop the extra dimension
        aa = self.fc_attention(xx.view(-1, 2*self.D2))[..., 0]
        # unflatten and normalize attention weights for each node
        return F.softmax(aa.view(N, K, K), dim=2)

    def forward(self, x):
        """ Apply the fully connected graph attention network layer

        Takes in a set of nodes from a fully connected graph. Applies a
        graph attention network layer. The set of nodes is passed in as a
        [N x K x D] tensor. N batches, K nodes, D dimension at each node.

        Arguments:
            x {torch.Tensor} -- [N x K x D1] tensor of node features

        Returns:
            torch.Tensor -- [N x K x D2] tensor of node features
        """
        N, K, _ = x.shape
        # apply the weight matrix to the node features
        x = self.W(x.view(-1, self.D1)).view(N, K, self.D2)
        # get an attention mask for each node
        a = self.attention(x)
        # apply the attention mask to the nodes features
        x = torch.einsum('nkj, nkd -> nkd', a, x)
        # and apply a nonlinearity
        return F.relu(x)
