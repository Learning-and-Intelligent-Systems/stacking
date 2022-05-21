import argparse

from matplotlib.pyplot import get
from learning.active import acquire
import torch
import numpy as np
from learning.models import latent_ensemble
from learning.active.utils import ActiveExperimentLogger
from particle_belief import GraspingDiscreteLikelihoodParticleBelief
from learning.domains.grasping.active_utils import get_fit_object, sample_unlabeled_data, get_labels, get_train_and_fit_objects
from learning.active.acquire import bald
# from learning.evaluate.planner import EnsemblePlanner
import sys

def particle_bald(predictions, weights, eps=1e-5):
    """ Get the BALD score for each example.
    :param predictions: (N, K) predictions for N datapoints from K models.
    :return: (N,) The BALD score for each of the datapoints.
    """
    predictions = predictions.cpu().numpy()
    norm = weights.sum()

    mp_c1 = np.sum(weights*predictions, axis=1)/norm
    mp_c0 = 1 - mp_c1

    m_ent = -(mp_c1 * np.log(mp_c1+eps) + mp_c0 * np.log(mp_c0+eps))

    p_c1 = predictions
    p_c0 = 1 - predictions

    ent_per_model = p_c1 * np.log(p_c1+eps) + p_c0 * np.log(p_c0+eps)
    ent = np.sum(ent_per_model*weights, axis=1)/norm

    bald = m_ent + ent
    return bald

def find_informative_tower(pf, object_set, logger, args):
    data_sampler_fn = lambda n: sample_unlabeled_data(n_samples=n, object_set=object_set)

    all_grasps = []
    all_preds = []
    for ix in range(0, args.n_samples):
        grasp_data = data_sampler_fn(1)
        preds = pf.get_particle_likelihoods(pf.particles.particles, grasp_data)
        all_preds.append(preds)
        all_grasps.append(grasp_data)
    
    pred_vec = torch.Tensor(np.stack(all_preds))
    # scores = bald(pred_vec).cpu().numpy()
    scores = particle_bald(pred_vec, pf.particles.weights)
    print('Scores:', scores)
    acquire_ix = np.argsort(scores)[::-1][0]

    return all_grasps[acquire_ix]

def particle_filter_loop(pf, object_set, logger, strategy, args):
    logger.save_ensemble(pf.likelihood, 0, symlink_tx0=True)
    logger.save_particles(pf.particles, 0)

    for tx in range(0, args.max_acquisitions):
        print('[ParticleFilter] Interaction Number', tx)
        
        # Choose a tower to build that includes the new block.
        if strategy == 'random':
            data_sampler_fn = lambda n: sample_unlabeled_data(n_samples=n, object_set=object_set)
            grasp_dataset = data_sampler_fn(1)
        elif strategy == 'bald':
            grasp_dataset = find_informative_tower(pf, object_set, logger, args)
        else:
            raise NotImplementedError()

        print(grasp_dataset['grasp_data']['object_ids'])
        # Get the observation for the chosen tower.
        grasp_dataset = get_labels(grasp_dataset)

        # Update the particle belief.
        particles, means = pf.update(grasp_dataset)

        # TODO: Save the model and particle distribution at each step.
        logger.save_acquisition_data(grasp_dataset, None, tx+1)
        logger.save_ensemble(pf.likelihood, tx+1, symlink_tx0=True)
        logger.save_particles(particles, tx+1)

# "grasp_train-ycb-test-ycb-1_fit_random_train_geo_object0": "learning/experiments/logs/grasp_train-ycb-test-ycb-1_fit_random_train_geo_object0-20220504-134253"
def run_particle_filter_fitting(args):
    print(args)
    args.use_latents = True
    args.fit_pf = True
    
    logger = ActiveExperimentLogger.setup_experiment_directory(args)

    # ----- Load the block set -----
    print('Loading objects:', args.objects_fname)
    object_set = get_train_and_fit_objects(pretrained_ensemble_path=args.pretrained_ensemble_exp_path,
                                           use_latents=True,
                                           fit_objects_fname=args.objects_fname,
                                           fit_object_ix=args.eval_object_ix)
    print('Total objects:', len(object_set['object_names']))
    args.num_eval_objects = 1
    args.num_train_objects = len(object_set['object_names']) - args.num_eval_objects

    # ----- Likelihood Model -----
    train_logger = ActiveExperimentLogger(exp_path=args.pretrained_ensemble_exp_path, use_latents=True)
    latent_ensemble = train_logger.get_ensemble(args.ensemble_tx)
    if torch.cuda.is_available():
        latent_ensemble.cuda()
    latent_ensemble.add_latents(1)

    # ----- Initialize particle filter from prior -----
    pf = GraspingDiscreteLikelihoodParticleBelief(
        object_set=object_set,
        D=latent_ensemble.d_latents,
        N=args.n_particles,
        likelihood=latent_ensemble,
        resample=False,
        plot=True)

    # ----- Run particle filter loop -----
    particle_filter_loop(pf, object_set, logger, args.strategy, args)
    
    return logger.exp_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default='', help='Where results will be saved. Randon number if not specified.')
    parser.add_argument('--max-acquisitions', type=int, default=25, help='Number of iterations to run the main active learning loop for.')
    parser.add_argument('--objects-fname', type=str, default='', help='File containing a list of objects to grasp.')
    parser.add_argument('--n-samples', type=int, default=1000)
    parser.add_argument('--pretrained-ensemble-exp-path', type=str, default='', help='Path to a trained ensemble.')
    parser.add_argument('--ensemble-tx', type=int, default=-1, help='Timestep of the trained ensemble to evaluate.')
    parser.add_argument('--eval-object-ix', type=int, default=0, help='Index of which eval object to use.')
    parser.add_argument('--strategy', type=str, choices=['bald', 'random', 'task'], default='bald')
    parser.add_argument('--n-particles', type=int, default=100)
    args = parser.parse_args()
    
    run_particle_filter_fitting(args)
    
