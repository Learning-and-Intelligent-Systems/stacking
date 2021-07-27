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
    model_paths = ['learning/experiments/logs/models/2_random-20210727-093617',
                    'learning/experiments/logs/models/3_random-20210727-100817',
                	'learning/experiments/logs/models/4_random-20210727-100851',
                    'learning/experiments/logs/models/5_random-20210727-100337']

    # get test dataset from logger
    expert_test_dataset_paths = ['learning/experiments/logs/datasets/2_expert-20210727-093318',
                                    'learning/experiments/logs/datasets/3_expert-20210727-093328',
                                    'learning/experiments/logs/datasets/4_expert-20210727-093335',
                                    'learning/experiments/logs/datasets/5_expert-20210727-093342',
                                    'learning/experiments/logs/datasets/6_expert-20210727-093400',
                                    'learning/experiments/logs/datasets/7_expert-20210727-093407',
                                    'learning/experiments/logs/datasets/8_expert-20210727-093415']

    random_test_dataset_paths = ['learning/experiments/logs/datasets/2_random_test-20210727-093504',
                                    'learning/experiments/logs/datasets/3_random_test-20210727-093459',
                                    'learning/experiments/logs/datasets/4_random_test-20210727-093452',
                                    'learning/experiments/logs/datasets/5_random_test-20210727-093446',
                                    'learning/experiments/logs/datasets/6_random_test-20210727-093440',
                                    'learning/experiments/logs/datasets/7_random_test-20210727-093433',
                                    'learning/experiments/logs/datasets/8_random_test-20210727-093426']

    test_dataset_paths = expert_test_dataset_paths
    #test_dataset_paths = random_test_dataset_paths

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
            assert test_dataset_path != train_dataset_path, 'Test dataset cannot be same as training dataset'
            test_dataset_logger = GoalConditionedExperimentLogger(test_dataset_path)
            test_trans_dataset = test_dataset_logger.load_trans_dataset()
            test_trans_dataset.set_pred_type(trans_model.pred_type)
            test_heur_dataset = test_dataset_logger.load_heur_dataset()
            test_dataset_num_blocks = test_dataset_logger.args.num_blocks
            accuracies['transition'][train_num_blocks][0].append(test_dataset_num_blocks)
            accuracies['transition'][train_num_blocks][1].append(calc_accuracy(trans_model, test_trans_dataset, 'transition'))
            accuracies['heuristic'][train_num_blocks][0].append(test_dataset_num_blocks)
            accuracies['heuristic'][train_num_blocks][1].append(calc_accuracy(heur_model, test_heur_dataset, 'heuristic'))

    # plot results
    logger = GoalConditionedExperimentLogger.setup_experiment_directory(args, 'accuracy')
    plot_accuracies(accuracies['transition'], 'Learned Transition Model Accuracy')
    plt.savefig(logger.exp_path+'/transition_accuracy.png')
    plot_accuracies(accuracies['heuristic'], 'Learned Heuristic Model Accuracy')
    plt.savefig(logger.exp_path+'/heuristic_accuracy.png')
    logger.save_plot_data([model_paths, test_dataset_paths, accuracies]) # TODO save_accuracy data?
    print('Saving figures and data to %s.' % logger.exp_path)
