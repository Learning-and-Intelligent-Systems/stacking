import argparse
import torch
from torch.utils.data import DataLoader
from copy import copy
from torch.nn import functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

from learning.active.train import train
from learning.active.utils import GoalConditionedExperimentLogger
from learning.models.goal_conditioned import TransitionGNN, HeuristicGNN
from learning.domains.abc_blocks.abc_blocks_data import ABCBlocksTransDataset, ABCBlocksHeurDataset
from visualize.domains.abc_blocks.performance import * #calc_trans_error_rate, calc_heur_error_rate, \
                                                        #vis_trans_errors, vis_trans_dataset_grid, \
                                                        #vis_trans_dataset_hist, \
                                                        #calc_successful_action_error_rate

'''
def evaluate(args, trans_model, train_trans_dataset, test_trans_dataset, train_world):
    print('Training Dataset')
    perc_t_explored = action_space_stats(train_world.num_objects, train_world.num_blocks, train_trans_dataset)
    train_accuracy = detailed_error_stats(args, train_trans_dataset, trans_model)
    print('Test Dataset')
    test_accuracy = detailed_error_stats(args, test_trans_dataset, trans_model)

    if args.plot:
        vis_trans_errors(args, test_trans_dataset, trans_model)
        vis_trans_dataset_grid(args, train_trans_dataset, 'Frequency of Edges seen in Training Dataset (n=%i)' % len(train_trans_dataset))
        vis_trans_dataset_grid(args, test_trans_dataset, 'Frequency of Edges seen in Test Dataset')
        vis_trans_dataset_hist(args, train_trans_dataset, 'Tower Heights in Training Data')
        vis_trans_dataset_hist(args, test_trans_dataset, 'Tower Heights in Test Data')
        plt.show()
    return perc_t_explored, test_accuracy
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',
                        action='store_true',
                        help='set to run in debug mode')
    #parser.add_argument('--goals-file-path',
    #                    type=str,
    #                    default='learning/domains/abc_blocks/goal_files/goals_05052021.csv',
    #                    help='path to csv file of goals to attempt')
    parser.add_argument('--pred-type',
                        type=str,
                        choices=['delta_state', 'full_state'],
                        default='delta_state')
    parser.add_argument('--exp-name',
                        type=str,
                        required=True,
                        help='path to save dataset to')
    parser.add_argument('--dataset-exp-path',
                        type=str,
                        required=True,
                        help='path to folder with training dataset')
    parser.add_argument('--batch-size',
                        type=int,
                        default=16,
                        help='training batch size')
    parser.add_argument('--n-epochs',
                        type=int,
                        default=300,
                        help='training epochs')
    parser.add_argument('--n-hidden',
                        type=int,
                        default=16,
                        help='number of hidden units in network')

    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    #try:
    n_of_in=1
    n_ef_in=1
    n_af_in=2

    dataset_logger = GoalConditionedExperimentLogger(args.dataset_exp_path)
    trans_dataset = dataset_logger.load_dataset()

    model_logger = GoalConditionedExperimentLogger.setup_experiment_directory(args, 'models')

    print('Training dataset %s.' % dataset_logger.exp_path)
    print('Training with %i datapoints.' % len(trans_dataset))
    trans_dataloader = DataLoader(trans_dataset, batch_size=args.batch_size, shuffle=False) # TODO: switch to shuffle?
    trans_model = TransitionGNN(args,
                                n_of_in=n_of_in,
                                n_ef_in=n_ef_in,
                                n_af_in=n_af_in,
                                n_hidden=args.n_hidden)
    if args.pred_type == 'delta_state':
        loss_fn = F.mse_loss
    elif args.pred_type == 'full_state':
        loss_fn = F.binary_cross_entropy
    train(trans_dataloader, None, trans_model, n_epochs=args.n_epochs, loss_fn=loss_fn)

    model_logger.save_model(trans_model)
    print('Saved model to %s.' % model_logger.exp_path)
    #except:
    #    import pdb; pdb.post_mortem()
