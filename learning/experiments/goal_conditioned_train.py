import argparse
import torch
from torch.utils.data import DataLoader
from copy import copy
from torch.nn import functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

from learning.active.train import train
from learning.domains.abc_blocks.world import ABCBlocksWorldGT
from learning.active.utils import ActiveExperimentLogger
from learning.models.goal_conditioned import TransitionGNN, HeuristicGNN
from learning.domains.abc_blocks.abc_blocks_data import ABCBlocksTransDataset, ABCBlocksHeurDataset
from learning.domains.abc_blocks.generate_data import generate_dataset, preprocess
from visualize.domains.abc_blocks.performance import * #calc_trans_error_rate, calc_heur_error_rate, \
                                                        #vis_trans_errors, vis_trans_dataset_grid, \
                                                        #vis_trans_dataset_hist, \
                                                        #calc_successful_action_error_rate

batch_size = 16
n_epochs = 300
n_hidden = 16
n_of_in=1
n_ef_in=1
n_af_in=2

def generate_world_dataset(args, num_blocks, mode):
    world = ABCBlocksWorldGT(num_blocks)
    trans_dataset = ABCBlocksTransDataset()
    heur_dataset = ABCBlocksHeurDataset()
    if mode == 'train':
        policy = world.random_policy
    elif mode == 'test':
        policy = world.expert_policy
    generate_dataset(args, world, None, trans_dataset, heur_dataset, policy)
    #preprocess(args, trans_dataset, type='balanced_actions')
    return trans_dataset, world

def setup_and_train(args, trans_dataset):
    print('Training with %i datapoints.' % len(trans_dataset))
    trans_dataloader = DataLoader(trans_dataset, batch_size=batch_size, shuffle=False)
    trans_model = TransitionGNN(args,
                                n_of_in=n_of_in,
                                n_ef_in=n_ef_in,
                                n_af_in=n_af_in,
                                n_hidden=n_hidden)
    if args.pred_type == 'delta_state':
        loss_fn = F.mse_loss
    elif args.pred_type == 'full_state':
        loss_fn = F.binary_cross_entropy
    train(trans_dataloader, None, trans_model, n_epochs=n_epochs, loss_fn=loss_fn)
    return trans_model

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',
                        action='store_true',
                        help='set to run in debug mode')
    parser.add_argument('--domain',
                        type=str,
                        choices=['abc_blocks', 'com_blocks'],
                        default='abc_blocks',
                        help='domain to generate data from')
    parser.add_argument('--goals-file-path',
                        type=str,
                        default='learning/domains/abc_blocks/goal_files/goals_05052021.csv',
                        help='path to csv file of goals to attempt')
    parser.add_argument('--max-seq-attempts',
                        type=int,
                        default=10,
                        help='max number of times to attempt to reach a given goal')
    parser.add_argument('--max-action-attempts',
                        type=int,
                        default=30,
                        help='max number of actions in a sequence')
    parser.add_argument('--pred-type',
                        type=str,
                        choices=['delta_state', 'full_state'],
                        default='delta_state')
    parser.add_argument('--exp-name',
                        type=str,
                        required=True,
                        help='path to save dataset to')
    parser.add_argument('--plot',
                        action='store_true')
    parser.add_argument('--train-num-blocks',
                        type=int,
                        default=3)
    parser.add_argument('--test-num-blocks',
                        type=int,
                        default=4)
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    #try:
    print('Generating test dataset.')
    test_trans_dataset, test_world = generate_world_dataset(args, args.test_num_blocks, 'test')

    print('Generating train dataset.')
    train_trans_dataset, train_world = generate_world_dataset(args, args.train_num_blocks, 'train')

    trans_model = setup_and_train(args, train_trans_dataset)

    evaluate(args, trans_model, train_trans_dataset, test_trans_dataset, train_world)

    #except:
    #    import pdb; pdb.post_mortem()
