import numpy as np
from torch import nn
from scipy.stats import norm

from learning.domains.throwing.throwing_data import generate_dataset, xs_to_actions, label_actions
from learning.domains.throwing.train_latent import get_predictions

def eval_hit_target(latent_ensemble, objects, data_pred_fn, n_samples=10000, n_targets=10):
	# sample a bunch of (xs, z_ids) and forward pass them to get predictions
	candidate_plans = generate_dataset(objects, n_samples, label=False)
	mus, sigmas = data_pred_fn(latent_ensemble, candidate_plans)

	# targets come from the same distribution
	_, _, targets = generate_dataset(objects, n_targets, label=True)
	
	# compute the likelihood of hitting each target under each sample
	plan_scores = norm.pdf(targets[None, :], loc=mus, scale=sigmas)

	# get the best plans indices
	best_plan_idxs = np.argmax(plan_scores, axis=0)
	actions = xs_to_actions(candidate_plans[0][best_plan_idxs])
	z_ids = candidate_plans[1][best_plan_idxs]

	# forward simulate to get the outcome of each plan
	ys = label_actions(objects, actions, z_ids)

	# score the outcomes
	score = np.abs(targets - ys).mean()
	return score
