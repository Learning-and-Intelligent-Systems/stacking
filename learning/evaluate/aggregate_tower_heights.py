import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt

from learning.experiments.generate_data import generate_dataset
from learning.active.utils import GoalConditionedExperimentLogger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',
                        action='store_true',
                        help='set to run in debug mode')

    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    data_paths = ['learning/experiments/logs/datasets/bar_plot-20210809-152040',
                    'learning/experiments/logs/datasets/bar_plot-20210809-152045',
                    'learning/experiments/logs/datasets/bar_plot-20210809-152049',
                    'learning/experiments/logs/datasets/bar_plot-20210809-152056',
                    'learning/experiments/logs/datasets/bar_plot-20210809-152058',
                    'learning/experiments/logs/datasets/bar_plot-20210809-152335',
                    'learning/experiments/logs/datasets/bar_plot-20210809-152336',
                    'learning/experiments/logs/datasets/bar_plot-20210809-152337',
                    'learning/experiments/logs/datasets/bar_plot-20210809-152339',
                    'learning/experiments/logs/datasets/bar_plot-20210809-152340']

    N = len(data_paths)
    # get num blocks from one data_path (should be the same value for all data_paths)
    logger = GoalConditionedExperimentLogger(data_paths[0])
    data_args = logger.load_args()
    num_blocks = data_args.num_blocks
    policy = data_args.policy
    count_values = np.zeros((num_blocks+1,N))
    for n, data_path in enumerate(data_paths):
        logger = GoalConditionedExperimentLogger(data_path)
        final_states = logger.load_final_states()
        for final_state in final_states:
            tower_height = len(final_state.stacked_blocks)
            count_values[tower_height][n] += 1
    mean = np.mean(count_values, axis=1)
    std = np.std(count_values, axis=1)
    plt.bar(range(num_blocks+1), mean, width=1., color='r', yerr=std)
    plt.title('Average Number of Tower Heights Generated over \n N=%i Runs with %i Blocks with %s Policy' % (N, num_blocks, policy))
    plt.xlabel('Tower Height')
    plt.ylabel('Mean Frequency (w/ StDev)')
    plt.show()
