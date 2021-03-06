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
     * D1 > D2

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
        self.W = nn.Sequential(
            nn.Linear(D1, D1),
            nn.LeakyReLU(),
            nn.Linear(D1, D2),
            nn.LeakyReLU()
        )

        # attention weights
        self.A = nn.Sequential(

            nn.Linear(2*D2, 2*D2),
            nn.LeakyReLU(),
            nn.Linear(2*D2, 1),
            nn.LeakyReLU()
        )

        # output layer
        self.fc_output = nn.Linear(D2, 1)

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
        aa = self.A(xx.view(-1, 2*self.D2))[..., 0]
        # unflatten and normalize attention weights for each node
        return F.softmax(aa.view(N, K, K), dim=2)

    def step(self, x):
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
        x_old = x[...,-self.D2:]

        # apply the weight matrix to the node features
        x = self.W(x.view(-1, self.D1)).view(N, K, self.D2)
        # get an attention mask for each node
        a = self.attention(x)
        # apply the attention mask to the nodes features
        x = torch.einsum('nkj, nkd -> nkd', a, x)
        # and finally a skip connection to make it a resnet
        x += x_old

        return x

    def output(self, x):
        N, K, _ = x.shape
        x = self.fc_output(x.view(-1, self.D2))
        x = x.view(N,K,1)[...,0]
        x = torch.sigmoid(x)
        return x.prod(axis=1)

    def forward(self, towers, k=None, x=None):
        N, K, _ = towers.shape

        # create additional channels to be used in the processing of the tower
        if x is None:
            x = 1e-2*torch.randn(N, K, self.D2)
            if torch.cuda.is_available():
                x = x.cuda()
        # run the network as many times as there are blocks in the tower
        if k is None:
            k = K

        # run the network as many times as there are blocks
        for _ in range(k):
            # append the tower information
            x = torch.cat([towers, x], axis=2)
            x = self.step(x)

        return self.output(x)
