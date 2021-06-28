import argparse
import torch
from torch.utils.data import DataLoader
from copy import copy
from torch.nn import functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

from learning.active.train import train
from learning.domains.abc_blocks.world import ABCBlocksWorld, MAX_OBJECTS
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
train_num_blocks = 2
test_num_blocks = 3

def run_goal_directed_train(train_args):
    print('Generating test dataset.')
    test_world = ABCBlocksWorld(test_num_blocks)
    test_args = copy(train_args)
    test_args.max_seq_attempts = 10
    test_args.exp_name = 'test1'
    test_trans_dataset = ABCBlocksTransDataset()
    test_heur_dataset = ABCBlocksHeurDataset()
    test_logger = ActiveExperimentLogger.setup_experiment_directory(test_args)
    test_policy = test_world.random_policy
    generate_dataset(test_args, test_world, test_logger, test_trans_dataset, test_heur_dataset, test_policy)
    #preprocess(test_args, test_trans_dataset, type='balanced_actions')
    
    train_world = ABCBlocksWorld(train_num_blocks)
    train_trans_dataset = ABCBlocksTransDataset()
    train_heur_dataset = ABCBlocksHeurDataset()
    train_logger = ActiveExperimentLogger.setup_experiment_directory(args)
    train_trans_dataloader = DataLoader(train_trans_dataset, batch_size=batch_size, shuffle=False)
    train_heur_dataloader = DataLoader(train_heur_dataset, batch_size=batch_size, shuffle=False)
    
    # add more sequences to dataset each time but retrain from scratch
    n_datapoints = []
    train_policy = train_world.random_policy
    for i, max_additional_seq_attempts in enumerate([10]):#[1, 4, 5, 10, 30, 50]):
        print('Adding to training dataset.')
        args.max_seq_attempts = max_additional_seq_attempts
        generate_dataset(train_args, train_world, train_logger, train_trans_dataset, train_heur_dataset, train_policy)
        #preprocess(train_args, train_trans_dataset, type='balanced_actions')
        train_trans_model = TransitionGNN(train_args, n_of_in=MAX_OBJECTS, n_ef_in=1, n_af_in=2*MAX_OBJECTS, n_hidden=n_hidden)
        print('Training with %i datapoints.' % len(train_trans_dataset))
        n_datapoints.append(len(train_trans_dataset))
        if args.pred_type == 'delta_state':
            loss_fn = F.mse_loss
        elif args.pred_type == 'full_state':
            loss_fn = F.binary_cross_entropy
        train(train_trans_dataloader, None, train_trans_model, n_epochs=n_epochs, loss_fn=loss_fn)
        print('training accuracy')
        detailed_error_stats(train_args, train_trans_dataset, train_trans_model)
        print('test accuracy')
        detailed_error_stats(train_args, test_trans_dataset, train_trans_model)
        if args.plot:
            vis_trans_errors(train_args, test_trans_dataset, train_trans_model)
            vis_trans_dataset_grid(train_args, train_trans_dataset, 'Frequency of Edges seen in Training Dataset (n=%i)' % len(trans_dataset))
            vis_trans_dataset_grid(test_args, test_trans_dataset, 'Frequency of Edges seen in Test Dataset')
            vis_trans_dataset_hist(train_args, train_trans_dataset, 'Tower Heights in Training Data')
            vis_trans_dataset_hist(test_args, test_trans_dataset, 'Tower Heights in Test Data')
        
        '''
        heur_model = HeuristicGNN()
        train(heur_dataloader, None, heur_model, n_epochs=100, loss_fn=F.mse_loss)
        heur_error_rate = calc_heur_error_rate(test_heur_dataset, heur_model)
        heur_error_rates.append(heur_error_rate)
        print('Heuristic Prediction Error Rate: %f' % heur_error_rate)
        '''

    '''
    if args.plot:
        # Visualize Error Rate
        fig, ax = plt.subplots()
        ax.plot(n_datapoints, trans_error_rates, '*-')
        ax.set_xlabel('Training Dataset Size (Number of Actions)')
        ax.set_ylabel('Edge Prediction Error Rate')
    '''
    if args.plot:
        plt.ion()
        plt.show()
        input('Enter to close plots.')
        plt.close()
    
    return n_datapoints#, trans_error_rates
    
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
                        default=10,
                        help='max number of actions in a sequence')
    parser.add_argument('--pred-type',
                        type=str,
                        choices=['delta_state', 'full_state'],
                        required=True)
    parser.add_argument('--exp-name',
                        type=str,
                        required=True,
                        help='path to save dataset to')
    parser.add_argument('--plot',
                        action='store_true')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()
        
    run_goal_directed_train(args)
    
    '''
    try:
        run_goal_directed_train(args)
    except:
        import pdb; pdb.post_mortem()
    '''