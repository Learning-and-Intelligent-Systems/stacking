from copy import deepcopy
import numpy as np
import torch
from torch import nn
from scipy.spatial.transform import Rotation


class LatentEnsemble(nn.Module):
    def __init__(self, ensemble, n_latents, d_latents, disable_latents=False):
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
        self.disable_latents = disable_latents

    def reset(self, random_latents=False):
        self.reset_latents(random=random_latents)
        self.ensemble.reset()
        # send to GPU if needed
        if torch.cuda.is_available():
            self.ensemble = self.ensemble.cuda()

    def reset_latents(self, random=False):
        with torch.no_grad():
            if random:
                self.latent_locs[:] = torch.randn_like(self.latent_locs)
                self.latent_logscales[:] = torch.randn_like(self.latent_logscales)
            else:
                self.latent_locs[:] = 0.
                self.latent_logscales[:] = 0.

    def change_number_of_latents(self, n_latents):
        """ copy the latent_ensemble, but change the number of latents """
        print(f"Changing the number of latents from {self.n_latents} to {n_latents}")
        d_latents = self.d_latents
        ensemble = deepcopy(self.ensemble)
        return self.__class__(ensemble, n_latents, d_latents)

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


    def forward(self, towers, block_ids, ensemble_idx=None, N_samples=1, collapse_latents=True, collapse_ensemble=True):
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

        # samples_for_each_tower_in_batch will be [N_batch x N_samples x tower_height x latent_dim]

        # Draw one sample for each block each time it appears in a tower
        q_z = torch.distributions.normal.Normal(self.latent_locs[block_ids],
                                                torch.exp(self.latent_logscales[block_ids]))
        samples_for_each_tower_in_batch = q_z.rsample(sample_shape=[N_samples]).permute(1, 0, 2, 3)

        samples_for_each_tower_in_batch = self.prerotate_latent_samples(towers, samples_for_each_tower_in_batch)
        towers_with_latents = self.concat_samples(
            samples_for_each_tower_in_batch, towers)

        # reshape the resulting tensor so the batch dimension holds
        # N_batch times N_samples
        N_batch, N_samples, N_blocks, total_dim = towers_with_latents.shape
        towers_with_latents = towers_with_latents.view(-1, N_blocks, total_dim)

        # forward pass of the model(s)
        if ensemble_idx is None:
            # prediction for each model in the ensemble ensemble
            # [(N_batch*N_samples) x N_ensemble]
            labels = self.ensemble.forward(towers_with_latents)
            labels = labels.view(N_batch, N_samples, -1).permute(0, 2, 1)
        else:
            # prediction of a single model in the ensemble
            labels = self.ensemble.models[ensemble_idx].forward(towers_with_latents)
            labels = labels[:, None, :]
            labels = labels.view(N_batch, N_samples, -1).permute(0, 2, 1)

        # N_batch x N_ensemble x N_samples
        if collapse_ensemble:
            labels = labels.mean(axis=1, keepdim=True)
        if collapse_latents:
            labels = labels.mean(axis=2, keepdim=True)

        return labels


class ThrowingLatentEnsemble(LatentEnsemble):

    def concat_samples(self, x, z_samples):
        """
        Arguments:
            x {torch.Tensor} -- [N_batch x D_observed]
            z_samples {torch.Tensor} -- [N_batch x N_samples x D_latent]

        Returns:
            [N_batch x N_samples x (D_observed+D_latent)]
        """
        N_batch, N_samples, D_latent = z_samples.shape
        x = x.unsqueeze(1).expand(-1, N_samples, -1)
        return torch.cat([x, z_samples], 2)

    def sample_latents(self, obj_ids, N_samples):
        # parameters will have shape [N_batch x D_latent]
        q_z = torch.distributions.normal.Normal(self.latent_locs[obj_ids],
                                                torch.exp(self.latent_logscales[obj_ids]))

        # if self.disable_latents: N_samples = 1 # NOTE(izzy): debugging

        # samples will have shape [N_batch x N_samples x D_latent]
        return q_z.rsample(sample_shape=[N_samples]).permute(1, 0, 2) #* float(not self.disable_latents)


    def forward(self, x, obj_ids, ensemble_idx=None, N_samples=1, collapse_latents=True, collapse_ensemble=True):
        assert x.shape[0] == obj_ids.shape[0], "One object per experiment"
        N_ensemble = self.ensemble.n_models

        # samples will have shape [N_batch x N_samples x D_latent]
        z_samples = self.sample_latents(obj_ids, N_samples)

        # data will have shape [N_batch x N_samples x (D_observed+D_latent)]
        x_with_z_samples = self.concat_samples(x, z_samples)

        # reshape the resulting tensor so the batch dimension holds
        # N_batch times N_samples
        N_batch, N_samples, D_total = x_with_z_samples.shape
        x_with_z_samples = x_with_z_samples.view(-1, D_total)

        if ensemble_idx is None:
            # prediction for each model in the ensemble ensemble
            # [(N_batch*N_samples) x N_ensemble x D_pred]
            labels = self.ensemble.forward(x_with_z_samples)
            labels = labels.view(N_batch, N_samples, N_ensemble, -1).permute(0, 2, 1, 3)
        else:
            # prediction of a single model in the ensemble
            # [(N_batch*N_samples) x D_pred]
            labels = self.ensemble.models[ensemble_idx].forward(x_with_z_samples)
            labels = labels[:, None, :]
            labels = labels.view(N_batch, N_samples, 1, -1).permute(0, 2, 1, 3)

        # N_batch x N_ensemble x N_samples
        if collapse_ensemble:
            labels = labels.mean(axis=1, keepdim=True)
        if collapse_latents:
            labels = labels.mean(axis=2, keepdim=True)

        return labels


class PFThrowingLatentEnsemble(ThrowingLatentEnsemble):
    def __init__(self, ensemble, n_latents, d_latents, n_particles=1000, disable_latents=False):
        """ Use a set of particles to represent the latent distribution, rather
        than a variation distribution (gaussian)

        Arguments:
            n_latents {int}: Number of blocks.
            d_latents {int}: Dimension of each latent.
        """
        super(PFThrowingLatentEnsemble, self).__init__(ensemble, n_latents, d_latents)

        self.ensemble = ensemble
        self.n_latents = n_latents
        self.d_latents = d_latents
        self.n_particles = n_particles

        # NOTE(izzy): so that reset works, we'll use the latent_locs
        # to store the particles. latent_scales are not neeced
        self.latent_locs = nn.Parameter(torch.randn(n_latents, n_particles, d_latents))
        self.latent_logscales = nn.Parameter(torch.zeros(0))
        self.disable_latents = disable_latents

        # this variable is used to determine the behavior when sampling from
        # the latent particles. when True, we don't sample because we want to
        # evaluate the likelihood of every particle
        self.fitting = False

    def reset_latents(self, random=False):
        # NOTE(izzy): the "random" argument is kept for consistency with other
        # LatentEnsemble classes, but we ignore it here
        with torch.no_grad():
            self.latent_locs[:] = torch.randn_like(self.latent_locs)

    def sample_latents(self, obj_ids, N_samples):
        # [N_batch x N_particles x D_latent]
        q_z = self.latent_locs[obj_ids]

        if self.fitting:
            return q_z
        else:
            # indices of particles
            idxs = np.random.randint(0, self.latent_locs.shape[1], size=N_samples)
            # pull out those particles
            return q_z[:, idxs]


def convert_to_particle_filter_latent_ensemble(latent_ensemble):
    ensemble = deepcopy(latent_ensemble.ensemble)
    return PFThrowingLatentEnsemble(ensemble,
                                    latent_ensemble.n_latents,
                                    latent_ensemble.d_latents)
