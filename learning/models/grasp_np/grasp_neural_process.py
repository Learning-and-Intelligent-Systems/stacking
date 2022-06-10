import numpy as np
import torch
import torch.nn as nn

from learning.models.pointnet import PointNetRegressor, PointNetClassifier, PointNetPerPointClassifier

class MultiTargetGraspNeuralProcess(nn.Module):

    def __init__(self, d_latents):
        super(MultiTargetGraspNeuralProcess, self).__init__()
        self.encoder = GNPEncoder(d_latents=d_latents)
        self.decoder = MultiTargetGNPDecoder(n_in=3+1+d_latents)
        self.d_latents = d_latents
        
    def forward(self, contexts, target_xs):
        mu, sigma = self.encoder(contexts)

        # Sample via reparameterization trick.
        q_z = torch.distributions.normal.Normal(mu, sigma)
        z = q_z.rsample()[:, :, None].expand(-1, -1, target_xs.shape[-1])
        
        # Replace True properties with latent samples.
        target_xs_with_latents = torch.cat([target_xs, z], dim=1)
                
        y_pred = self.decoder(target_xs_with_latents)
        return y_pred, q_z

class GraspNeuralProcess(nn.Module):

    def __init__(self, d_latents):
        super(GraspNeuralProcess, self).__init__()
        self.encoder = GNPEncoder(d_latents=d_latents)
        self.decoder = GNPDecoder(n_in=6+d_latents)
        self.d_latents = d_latents
        
    def forward(self, contexts, target_xs, n_context=-1):
        total_context_points = int(contexts[0, 3, :].sum().item())
        assert(n_context <= total_context_points)
        if n_context == -1:
            n_context = 49 #np.random.randint(total_context_points)
        
        # Only keep specified amount of context points.
        keep_ixs = torch.randperm(total_context_points)[:n_context]
        contexts = torch.cat([
            contexts[:, :, keep_ixs],
            contexts[:, :, total_context_points:]
        ], dim=2)

        mu, sigma = self.encoder(contexts)

        # Sample via reparameterization trick.
        q_z = torch.distributions.normal.Normal(mu, sigma)
        z = q_z.rsample()[:, :, None].expand(-1, -1, target_xs.shape[-1])
        
        # Replace True properties with latent samples.
        target_xs_with_latents = torch.cat([target_xs[:, :-5, :], z], dim=1)
                
        y_pred = self.decoder(target_xs_with_latents)
        return y_pred, q_z
        
class GNPEncoder(nn.Module):

    def __init__(self, d_latents):
        super(GNPEncoder, self).__init__()
        self.pointnet = PointNetRegressor(n_in=5, n_out=d_latents*2)
        self.d_latents = d_latents

    def forward(self, contexts):
        
        x = self.pointnet(contexts)
        mu, log_sigma = x[..., :self.d_latents], x[..., self.d_latents:]
        sigma = 0.01 + 0.99 * torch.sigmoid(log_sigma)
        #sigma = 0.01 + torch.exp(log_sigma)
        return mu, sigma   
        
class GNPDecoder(nn.Module):

    def __init__(self, n_in):
        super(GNPDecoder, self).__init__()
        self.pointnet = PointNetClassifier(n_in=n_in)

    def forward(self, target_xs):
        return self.pointnet(target_xs)

class MultiTargetGNPDecoder(nn.Module):

    def __init__(self, n_in):
        super(MultiTargetGNPDecoder, self).__init__()
        self.pointnet = PointNetPerPointClassifier(n_in=n_in)

    def forward(self, target_xs):
        return self.pointnet(target_xs)
