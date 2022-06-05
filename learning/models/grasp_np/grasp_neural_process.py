import numpy as np
import torch
import torch.nn as nn

from learning.models.pointnet import PointNetRegressor, PointNetClassifier


class GraspNeuralProcess(nn.Module):

    def __init__(self, d_latents):
        super(GraspNeuralProcess, self).__init__()
        self.encoder = GNPEncoder(d_latents=d_latents)
        self.decoder = GNPDecoder(n_in=6+d_latents)
        self.d_latents = d_latents
        
    def forward(self, contexts, target_xs, n_context=-1):
        total_context_points = contexts[:, 3].sum()
        assert(n_context < total_context_points)
        if n_context == -1:
            n_context = np.random.randint(total_context_points)
        
        # Only keep specified amount of context points.
        keep_ixs = torch.randperm(total_context_points)[:n_context]
        contexts = torch.cat([
            contexts[:, keep_ixs, :],
            contexts[:, total_context_points:, :]
        ], dim=1)

        mu, sigma = self.encoder(contexts)

        # Sample via reparameterization trick.
        q_z = torch.distributions.normal.Normal(mu, sigma)
        z = q_z.rsample()
        
        # Replace True properties with latent samples.
        target_xs = torch.cat([target_xs[:, :, :-self.d_latents], z], dim=2)

        y_pred = self.decoder(target_xs)
        return y_pred, q_z
        
class GNPEncoder(nn.Module):

    def __init__(self, d_latents):
        super(GNPEncoder, self).__init__()
        self.pointnet = PointNetRegressor(n_in=5, n_out=d_latents*2)
        self.d_latents = d_latents

    def forward(self, contexts):
        x = self.pointnet(contexts)
        mu, log_sigma = x[..., :self.d_latents], x[..., self.d_latents:]
        sigma = 0.1 + 0.9 + torch.sigmoid(log_sigma)
        return mu, sigma   
        
class GNPDecoder(nn.Module):

    def __init__(self, n_in):
        super(GNPEncoder, self).__init__()
        self.pointnet = PointNetClassifier(n_in=n_in)

    def forward(self, target_xs):
        return self.pointnet(target_xs)
