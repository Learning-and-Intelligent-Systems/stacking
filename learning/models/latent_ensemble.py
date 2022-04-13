import torch

from torch import nn
from scipy.spatial.transform import Rotation


class LatentEnsemble(nn.Module):
    def __init__(self, ensemble, n_latents, d_latents):
        """
        Arguments:
            n_latents {int}: Number of blocks.
            d_latents {int}: Dimension of each latent.
        """
        super(LatentEnsemble, self).__init__()

        self.ensemble = ensemble
        self.n_latents = n_latents
        self.d_latents = d_latents

        self.latent_locs = nn.Parameter(torch.zeros(n_latents, d_latents))
        self.latent_logscales = nn.Parameter(torch.zeros(n_latents, d_latents))

    def reset(self, random_latents=False):
        self.reset_latents(random=random_latents)
        self.ensemble.reset()
        # send to GPU if needed
        if torch.cuda.is_available():
            self.ensemble = self.ensemble.cuda()

    def reset_latents(self, ixs=[], random=False):
        """
        Reset all or a subset of the latent variables. 
        :param ixs: A list of which indices to reset.
        :param random: Set True to randomly reset latents, otherwise set to prior.
        """
        with torch.no_grad():
            if random:
                if len(ixs) == 0:
                    self.latent_locs[:] = torch.randn_like(self.latent_locs)
                    self.latent_logscales[:] = torch.randn_like(self.latent_logscales)
                else:
                    self.latent_locs[ixs, :] = torch.randn((len(ixs), self.d_latents))
                    self.latent_logscales[ixs, :] = torch.randn((len(ixs), self.d_latents))
            else:
                if len(ixs) == 0:
                    self.latent_locs[:] = 0.
                    self.latent_logscales[:] = 0.
                else:
                    self.latent_locs[ixs, :] = 0.
                    self.latent_logscales[ixs, :] = 0.

    def add_latents(self, n_latents):
        """
        Call this function when more blocks are added to increase the number of latent variables.
        :param n_latents: The number of latents to add.
        """
        self.n_latents += n_latents
        new_locs = torch.zeros(n_latents, self.d_latents)
        new_logscales = torch.zeros(n_latents, self.d_latents)
        if torch.cuda.is_available():
            new_locs = new_locs.cuda()
            new_logscales = new_logscales.cuda()

        self.latent_locs = nn.Parameter(torch.cat([self.latent_locs.data, new_locs], dim=0))
        self.latent_logscales = nn.Parameter(torch.cat([self.latent_logscales.data, new_logscales], dim=0))

    def associate(self, samples, block_ids):
        """ given samples from the latent space for each block in the set,
        reorder them so they line up with blocks in towers

        Arguments:
            samples {torch.Tensor} -- [N_samples x N_blockset x latent_dim]
            block_ids {torch.Tensor} -- [N_batch x N_blocks]

        Returns:
            torch.Tensor -- [N_batch x N_samples x N_blocks x latent_dim]
        """
        return samples[:, block_ids, :].permute(1, 0, 2, 3)

    def concat_samples(self, samples, observed):
        """ concatentate samples from the latent space for each tower
        with the observed variables for each tower

        Arguments:
            samples {torch.Tensor} -- [N_batch x N_samples x N_blocks x latent_dim]
            observed {torch.Tensor} -- [N_batch x N_blocks x observed_dim]

        Returns:
            torch.Tensor -- [N_batch x N_samples x N_blocks x total_dim]
        """
        N_batch, N_samples, N_blocks, latent_dim = samples.shape
        observed = observed.unsqueeze(1).expand(-1, N_samples, -1, -1)
        return torch.cat([samples, observed], 3)

    def prerotate_latent_samples(self, towers, samples):
        """ concatentate samples from the latent space for each tower
        with the observed variables for each tower

        Arguments:
            samples {torch.Tensor} -- [N_batch x N_samples x N_blocks x latent_dim]
            towers {torch.Tensor}   [N_batch x N_blocks x N_features]

        Returns:
            torch.Tensor -- [N_batch x N_samples x N_blocks x latent_dim]
        """
        N_batch, N_samples, N_blocks, latent_dim = samples.shape

        # pull out the quaternions for each block, and flatten the batch+block dims
        quats = towers[..., -4:].reshape(-1, 4)
        # create rotation matrices from the quaternions
        r = Rotation.from_quat(quats.cpu()).as_matrix()
        r = torch.Tensor(r)
        if torch.cuda.is_available(): r = r.cuda()
        # unflatten the batch+block dims and expand the sample dimension
        # now it should be [N_batch x N_samples x N_blocks x 3 x 3]
        r = r.view(N_batch, N_blocks, 3, 3).unsqueeze(1).expand(-1, N_samples, -1, -1, -1)
        # apply the rotation to the last three dimensions of the samples
        samples[...,-3:] = torch.einsum('asoij, asoj -> asoi', r, samples[...,-3:])
        # and return the result
        return samples

    def forward(self, towers, block_ids, ensemble_idx=None, N_samples=1, collapse_latents=True, collapse_ensemble=True, keep_latent_ix=-1, latent_samples=None, pf_latent_ix=-1):
        """ predict feasibility of the towers

        Arguments:
            towers {torch.Tensor}   [N_batch x N_blocks x N_features]
            block_ids {torch.Tensor}   [N_batch x N_blocks]

        Keyword Arguments:
            ensemble_idx {int} -- if None, average of all models (default: {None})
            N_samples {number} -- how many times to sample the latents (default: {1})

        Returns:
            torch.Tensor -- [N_batch]
        """
        N_batch, N_blocks, N_feats = towers.shape
        # samples_for_each_tower_in_batch will be [N_batch x N_samples x tower_height x latent_dim]
        # Draw one sample for each block each time it appears in a tower
        if keep_latent_ix < 0 and collapse_latents:
            q_z = torch.distributions.normal.Normal(self.latent_locs[block_ids],
                                                    torch.exp(self.latent_logscales[block_ids]))  # [N_batch, N_blocks, latent_dim]
            samples_for_each_tower_in_batch = q_z.rsample(sample_shape=[N_samples]).permute(1, 0, 2, 3)  # [N_batch, N_samples, N_blocks, latent_dim]
            if pf_latent_ix > -1:
                for tx in range(0, N_batch):
                    for bx_tower in range(0, N_blocks):
                        bx_blockset = block_ids[tx, bx_tower]
                        if bx_blockset == pf_latent_ix:
                            samples_for_each_tower_in_batch[tx, :, bx_tower, :] = latent_samples
            samples_for_each_tower_in_batch = self.prerotate_latent_samples(towers, samples_for_each_tower_in_batch)
            towers_with_latents = self.concat_samples(
                samples_for_each_tower_in_batch, towers)
        else:
            # Assume that keep_latent_ix is in each tower.
            
            # First create latents which are shape: [N_batch, N_samples (marg), N_samples (rest of latents), N_blocks, latent_dim]
            samples_for_each_tower_in_batch = torch.zeros((N_batch, N_samples, N_samples, N_blocks, self.d_latents))
            if torch.cuda.is_available():
                samples_for_each_tower_in_batch = samples_for_each_tower_in_batch.cuda()
            
            # Do one tower at a time. Might need to vectorize later if too slow.
            for tx in range(0, N_batch):
                for bx_tower in range(0, N_blocks):
                    bx_blockset = block_ids[tx, bx_tower]
                    q_z = torch.distributions.normal.Normal(self.latent_locs[bx_blockset],
                                                            torch.exp(self.latent_logscales[bx_blockset]))
                    if (bx_blockset == keep_latent_ix) and (latent_samples is None):
                        zk_samples = q_z.rsample(sample_shape=[N_samples])  # [N_samples, latent_dim]
                        samples_for_each_tower_in_batch[tx, :, :, bx_tower, :] = zk_samples.unsqueeze(1)
                    elif bx_blockset == keep_latent_ix:
                        samples_for_each_tower_in_batch[tx, :, :, bx_tower, :] = latent_samples.unsqueeze(1)
                    else:
                        zk_samples = q_z.rsample(sample_shape=[N_samples, N_samples])  # [N_samples, N_samples, latent_dim]
                        samples_for_each_tower_in_batch[tx, :, :, bx_tower, :] = zk_samples
            
            # Temporarily combine sample dimensions.
            samples_for_each_tower_in_batch = samples_for_each_tower_in_batch.view(N_batch, N_samples*N_samples, N_blocks, self.d_latents)
            samples_for_each_tower_in_batch = self.prerotate_latent_samples(towers, samples_for_each_tower_in_batch)
            towers_with_latents = self.concat_samples(samples_for_each_tower_in_batch, towers)


        # reshape the resulting tensor so the batch dimension holds
        # N_batch times N_samples
        N_batch, N_total_samples, N_blocks, total_dim = towers_with_latents.shape
        towers_with_latents = towers_with_latents.view(-1, N_blocks, total_dim)

        # forward pass of the model(s)
        if ensemble_idx is None:
            # prediction for each model in the ensemble ensemble
            # [(N_batch*N_samples) x N_ensemble]
            labels = self.ensemble.forward(towers_with_latents)
            labels = labels.view(N_batch, N_total_samples, -1).permute(0, 2, 1)
        else:
            # prediction of a single model in the ensemble
            labels = self.ensemble.models[ensemble_idx].forward(towers_with_latents)
            labels = labels[:, None, :]
            labels = labels.view(N_batch, N_total_samples, -1).permute(0, 2, 1)

        # N_batch x N_ensemble x N_samples
        if collapse_ensemble:
            labels = labels.mean(axis=1, keepdim=True)
        if collapse_latents:
            if keep_latent_ix < 0:
                labels = labels.mean(axis=2, keepdim=True)
            else:
                labels = labels.view(N_batch, -1, N_samples, N_samples)
                labels = labels.mean(axis=3)

        return labels


class GraspingLatentEnsemble(LatentEnsemble):

    def __init__(self, ensemble, n_latents, d_latents):
        super(GraspingLatentEnsemble, self).__init__(ensemble, n_latents, d_latents)

    def concat_samples(self, samples, observed):
        """
        :param samples: Per grasp LV samples (N_batch, N_samples, N_latent_dim).
        :param observed: Per grasp observed properties (N_batch, N_obs_dim, N_point).
        """
        N_batch, N_samples, N_latent_dim = samples.shape
        _, _, N_point = observed.shape

        samples = samples.unsqueeze(-1).expand(-1, -1, -1, N_point)
        observed = observed.unsqueeze(1).expand(-1, N_samples, -1, -1)
        
        return torch.cat([observed, samples], dim=2)

    def forward(self, X, object_ids, ensemble_idx=None, N_samples=1, collapse_latents=True, collapse_ensemble=True, keep_latent_ix=-1, latent_samples=None, pf_latent_ix=-1):
        """ Samples from LVs and concatenates each to the observed input before call the ensemble models.
        :param X: Observed grasping data of shape (batch_size, n_observed_dims, n_points)
        :param object_ids: List of object ids of shape (batch_size,)
        :param ensemble_idx: Whether to evaluate a specific ensemble ix or the entire ensemble.
        :param N_samples: Number of samples to take from the LVs.
        :param collapse_latents: Average prediction across latent samples. If False, will return predictions for all latent samples.
        :param collapse_ensemble: Average prediction across ensemble models. If False, will return predictions for all ensembles.
        :param keep_latent_ix: Not implemented.
        :param latent_samples: Not implemented.
        :param pf_latent_ix: Not implemented.
        """
        N_batch, N_observed_dims, N_points = X.shape

        if (pf_latent_ix > -1) and (latent_samples is not None):
            samples_for_each_grasp_in_batch = latent_samples.unsqueeze(0).expand(N_batch, N_samples, -1)
        else:
            q_z = torch.distributions.normal.Normal(
                self.latent_locs[object_ids],
                torch.exp(self.latent_logscales[object_ids]))  # [N_batch, latent_dim]
            samples_for_each_grasp_in_batch = q_z.rsample(sample_shape=[N_samples]).permute(1, 0, 2)  # [N_batch, N_samples, latent_dim]

        grasps_with_latents = self.concat_samples(samples_for_each_grasp_in_batch, X)

        # reshape the resulting tensor so the batch dimension holds
        # N_batch times N_samples
        N_batch, N_total_samples, total_dim, N_points = grasps_with_latents.shape
        grasps_with_latents = grasps_with_latents.view(-1, total_dim, N_points)

        # forward pass of the model(s)
        if ensemble_idx is None:
            # prediction for each model in the ensemble ensemble
            # [(N_batch*N_samples) x N_ensemble]
            labels = self.ensemble.forward(grasps_with_latents)
            labels = labels.view(N_batch, N_total_samples, -1).permute(0, 2, 1)
        else:
            # prediction of a single model in the ensemble
            labels = self.ensemble.models[ensemble_idx].forward(grasps_with_latents)
            labels = labels[:, None, :]
            labels = labels.view(N_batch, N_total_samples, -1).permute(0, 2, 1)

        # N_batch x N_ensemble x N_samples
        if collapse_ensemble:
            labels = labels.mean(axis=1, keepdim=True)
        if collapse_latents:
            if pf_latent_ix < 0:
                labels = labels.mean(axis=2, keepdim=True)
            else:
                labels = labels.view(N_batch, -1, N_samples)
                #labels = labels.mean(axis=3)

        return labels