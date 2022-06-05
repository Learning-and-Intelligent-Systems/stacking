import torch
import torch.nn as nn

from learning.models.pointnet import PointNetEncoder


class GraspNeuralProcess(nn.Module):

    def __init__(self):
        super(GraspNeuralProcess, self).__init__()
        self.encoder = GNPEncoder()
        self.decoder = GNPDecoder()
        
    def forward(self, context, targets):
        pass


class GNPEncoder(nn.Module):

    def __init__(self):
        super(GNPEncoder, self).__init__()

    def forward(self, context):
        pass


class GNPDecoder(nn.Module):

    def __init__(self):
        super(GNPEncoder, self).__init__()

    def forward(self):
        pass
