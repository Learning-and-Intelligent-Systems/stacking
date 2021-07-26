import argparse
import numpy as np
import matplotlib.pyplot as plt

from learning.active.utils import GoalConditionedExperimentLogger
from learning.domains.abc_blocks.abc_blocks_data import model_forward

def calc_accuracy(model, test_dataset, model_type):
    xs, ys = test_dataset[:]
    ys = ys.detach().numpy()
    preds = model_forward(model, xs)
    preds = preds.round()
    sum_axes = tuple(range(len(preds.shape)))[1:]
    accuracy = np.sum(np.all(ys == preds.round(), axis=sum_axes))/len(preds)
    return accuracy

def plot_accuracies(accuracies, title):
    fig, axis = plt.subplots()
    for train_num_blocks, model_accuracies in accuracies.items():
        axis.plot(model_accuracies[0], model_accuracies[1], label=train_num_blocks)
    axis.legend(title='Number of Training Blocks')
    axis.set_title(title)
    axis.set_ylabel('Accuracy')
    axis.set_xlabel('Number of Test Blocks')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name',
                        type=str,
                        required=True,
                        help='where to save exp data')
    parser.add_argument('--debug',
                        action='store_true',
                        help='set to run in debug mode')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    # get models from logger
    model_paths = ['learning/experiments/logs/models/2_random-20210719-150928',
                    'learning/experiments/logs/models/3_random-20210719-151007',
                	'learning/experiments/logs/models/4_random-20210719-151107',
                    'learning/experiments/logs/models/5_random-20210719-151247']

    # get test dataset from logger
    expert_test_dataset_paths = ['learning/experiments/logs/datasets/2_expert-20210719-150611',
                                    'learning/experiments/logs/datasets/3_expert-20210719-150622',
                                    'learning/experiments/logs/datasets/4_expert-20210719-150627',
                                    'learning/experiments/logs/datasets/5_expert-20210719-150634',
                                    'learning/experiments/logs/datasets/6_expert-20210719-150639',
                                    'learning/experiments/logs/datasets/7_expert-20210719-150644',
                                    'learning/experiments/logs/datasets/8_expert-20210719-150649']

    random_test_dataset_paths = ['learning/experiments/logs/datasets/2_random-20210719-150504',
                                    'learning/experiments/logs/datasets/3_random-20210719-150512',
                                    'learning/experiments/logs/datasets/4_random-20210719-150520',
                                    'learning/experiments/logs/datasets/5_random-20210719-150529',
                                    'learning/experiments/logs/datasets/6_random-20210719-152416',
                                    'learning/experiments/logs/datasets/7_random-20210719-152421',
                                    'learning/experiments/logs/datasets/8_random-20210719-152427']

    #test_dataset_paths = expert_test_dataset_paths
    test_dataset_paths = random_test_dataset_paths

    # calculate accuracy
    accuracies = {'transition': {}, 'heuristic': {}}
    for model_path in model_paths:
        model_logger = GoalConditionedExperimentLogger(model_path)
        train_dataset_path = model_logger.args.dataset_exp_path
        trans_model = model_logger.load_trans_model()
        heur_model = model_logger.load_heur_model()
        train_num_blocks = model_logger.args.num_blocks
        if not train_num_blocks in accuracies['transition']:
            accuracies['transition'][train_num_blocks] = [[], []]
            accuracies['heuristic'][train_num_blocks] = [[], []]
        for test_dataset_path in test_dataset_paths:
            assert (test_dataset_path != train_dataset_path, 'Cannot evaluate on training dataset')
            test_dataset_logger = GoalConditionedExperimentLogger(test_dataset_path)
            test_trans_dataset = test_dataset_logger.load_trans_dataset()
            test_heur_dataset = test_dataset_logger.load_heur_dataset()
            test_dataset_num_blocks = test_dataset_logger.args.num_blocks
            accuracies['transition'][train_num_blocks][0].append(test_dataset_num_blocks)
            accuracies['transition'][train_num_blocks][1].append(calc_accuracy(trans_model, test_trans_dataset, 'transition'))
            accuracies['heuristic'][train_num_blocks][0].append(test_dataset_num_blocks)
            accuracies['heuristic'][train_num_blocks][1].append(calc_accuracy(heur_model, test_heur_dataset, 'heuristic'))

    # plot results
    plot_accuracies(accuracies['transition'], 'Learned Transition Model Accuracy')
    plot_accuracies(accuracies['heuristic'], 'Learned Heuristic Model Accuracy')
    plt.show()
