import argparse
import matplotlib.pyplot as plt
import numpy as np


from learning.active.utils import ActiveExperimentLogger
from learning.evaluate.active_evaluate_towers import plot_latent_uncertainty
from learning.domains.throwing.task import eval_hit_target
from learning.domains.throwing.entities import ThrowingBall


def plot_task_performance(logger, task_score_fn, ax=plt.gca()):
	scores = []
	for tx in range(logger.args.max_acquisitions):
		latent_ensemble = logger.get_ensemble(tx)
		score = task_score_fn(latent_ensemble)
		print(f'Step {tx}. Score {score}')
		scores.append(score)

	ax.plot(scores)
	ax.set_xlabel('Acquisition Step')
	ax.set_ylabel('Task Loss (meters to target)')
	ax.set_title('Task Performance')
	np.save(logger.get_results_path('task_performance.npy'), np.array(scores))
	plt.savefig(logger.get_figure_path('task_performance.png'))
	plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-path', type=str, required=True)
    args = parser.parse_args()
    
    logger = ActiveExperimentLogger(args.exp_path, use_latents=True)
    logger.args.max_acquisitions = 50  # lazy
    logger.args.throwing = True # lazy

    # plot_latent_uncertainty(logger)

    objects = logger.get_objects(ThrowingBall)
    task_score_fn = lambda latent_ensemble: eval_hit_target(latent_ensemble, objects)
    plot_task_performance(logger, task_score_fn)
