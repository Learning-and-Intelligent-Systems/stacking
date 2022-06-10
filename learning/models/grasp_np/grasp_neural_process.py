import numpy as np
import torch
import torch.nn as nn

from learning.models.pointnet import PointNetEncoder, PointNetRegressor, PointNetClassifier, PointNetPerPointClassifier

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



class CustomGraspNeuralProcess(nn.Module):

    def __init__(self, d_latents):
        super(CustomGraspNeuralProcess, self).__init__()
        self.encoder = CustomGNPEncoder(d_latents=d_latents)
        self.decoder = CustomGNPDecoder(n_in=3+3+d_latents)
        self.d_latents = d_latents
        
    def forward(self, contexts, target_xs):
        mu, sigma = self.encoder(*contexts)

        # Sample via reparameterization trick.
        q_z = torch.distributions.normal.Normal(mu, sigma)

        # Replace True properties with latent samples.
        target_geoms, target_mids = target_xs
        n_batch, n_grasp, _, n_pts = target_geoms.shape
        z = q_z.rsample()[:, None, :, None].expand(n_batch, n_grasp, self.d_latents, n_pts)
        target_mids = target_mids[:, :, :, None].expand(n_batch, n_grasp, 3, n_pts)
        target_xs_with_latents = torch.cat([target_geoms, target_mids, z], dim=2)
                
        y_pred = self.decoder(target_xs_with_latents)
        return y_pred, q_z


class CustomGNPDecoder(nn.Module):

    def __init__(self, n_in):
        super(CustomGNPDecoder, self).__init__()
        self.pointnet = PointNetClassifier(n_in=n_in)

    def forward(self, target_xs):
        """
        :param target_xs: (batch_size, n_grasps, n_in, n_points)
        """
        n_batch, n_grasps, n_in, n_pts = target_xs.shape
        xs = target_xs.view(-1, n_in, n_pts)
        xs = self.pointnet(xs)
        return xs.view(n_batch, n_grasps, 1)


class CustomGNPEncoder(nn.Module):

    def __init__(self, d_latents):
        super(CustomGNPEncoder, self).__init__()

        # Used to encode local geometry.
        n_out_geom = 64
        self.pn_geom = PointNetRegressor(n_in=3, n_out=n_out_geom)
        self.pn_grasp = PointNetRegressor(n_in=3+1+n_out_geom, n_out=d_latents*2) # Input is grasp_midpoint, 
        self.d_latents = d_latents

    def forward(self, context_geoms, context_midpoints, context_labels):
        """
        :param context_geoms: (batch_size, n_grasps, 3, n_points)
        :param context_midpoints: (batch_size, n_grasps, 3)
        :param context_labels: (batch_size, n_grasps, 1)
        """
        n_batch, n_grasp, _, n_geom_pts = context_geoms.shape
        geoms = context_geoms.view(-1, 3, n_geom_pts)
        geoms_enc = self.pn_geom(geoms).view(n_batch, n_grasp, -1)

        grasp_input = torch.cat([context_midpoints, context_labels[:, :, None], geoms_enc], dim=2).swapaxes(1, 2)
        x = self.pn_grasp(grasp_input)
        mu, log_sigma = x[..., :self.d_latents], x[..., self.d_latents:]
        sigma = 0.01 + 0.99 * torch.sigmoid(log_sigma)
        #sigma = 0.01 + torch.exp(log_sigma)
        return mu, sigma   
