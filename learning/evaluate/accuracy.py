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
        axis.plot(model_accuracies[0], model_accuracies[1], '*', label=train_num_blocks)
    axis.legend(title='Number of Training Blocks')
    axis.set_title(title)
    axis.set_ylabel('Accuracy')
    axis.set_xlabel('Number of Test Blocks')
    axis.set_ylim(0.0, 1.0)

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
    model_paths = ['learning/experiments/logs/models/large-train-set-20210810-160955',
                    'learning/experiments/logs/models/large-train-set-20210810-161047',
                    'learning/experiments/logs/models/large-train-set-20210810-161118',
                    'learning/experiments/logs/models/large-train-set-20210810-161223']

    # Expert Test Sets
    expert_test_dataset_paths = ['learning/experiments/logs/datasets/large-test-set-2-20210810-161843',
        'learning/experiments/logs/datasets/large-test-set-3-20210810-161852',
        'learning/experiments/logs/datasets/large-test-set-4-20210810-161859',
        'learning/experiments/logs/datasets/large-test-set-5-20210810-161906',
        'learning/experiments/logs/datasets/large-test-set-6-20210810-161913']

    # Random Test Sets (2,3,4,5,6 num blocks)
    random_test_dataset_paths = ['learning/experiments/logs/datasets/large-test-set-random-2-20210810-165325',
        'learning/experiments/logs/datasets/large-test-set-random-3-20210810-165531',
        'learning/experiments/logs/datasets/large-test-set-random-4-20210810-165541',
        'learning/experiments/logs/datasets/large-test-set-random-5-20210810-165608',
        'learning/experiments/logs/datasets/large-test-set-random-6-20210810-165910']

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
