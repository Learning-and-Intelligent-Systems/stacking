import argparse
import torch
from torch.utils.data import DataLoader
from copy import copy
from torch.nn import functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

from learning.active.train import train
from learning.domains.abc_blocks.world import ABCBlocksWorld
from learning.active.utils import ActiveExperimentLogger
from learning.models.goal_conditioned import TransitionGNN, HeuristicGNN
from learning.domains.abc_blocks.abc_blocks_data import ABCBlocksTransDataset, ABCBlocksHeurDataset
from learning.domains.abc_blocks.generate_data import generate_dataset, preprocess
from visualize.domains.abc_blocks.performance import calc_trans_error_rate, calc_heur_error_rate, \
                                                        vis_trans_errors, vis_trans_dataset_grid, \
                                                        vis_trans_dataset_hist, \
                                                        calc_successful_action_error_rate

def run_goal_directed_train(args, plot=True):
    if args.domain == 'abc_blocks':
        world = ABCBlocksWorld()
        trans_dataset = ABCBlocksTransDataset()
        heur_dataset = ABCBlocksHeurDataset()

    logger = ActiveExperimentLogger.setup_experiment_directory(args)
    trans_dataloader = DataLoader(trans_dataset, batch_size=16, shuffle=False)
    heur_dataloader = DataLoader(heur_dataset, batch_size=16, shuffle=False)
        
    print('Generating test dataset.')
    test_args = copy(args)
    test_args.max_seq_attempts = 10
    test_args.exp_name = 'test1'
    test_trans_dataset = ABCBlocksTransDataset()
    test_heur_dataset = ABCBlocksHeurDataset()
    test_logger = ActiveExperimentLogger.setup_experiment_directory(test_args)
    test_policy = world.random_policy
    generate_dataset(test_args, world, test_logger, test_trans_dataset, test_heur_dataset, test_policy)
    
    # add more sequences to dataset each time but retrain from scratch
    n_datapoints = []
    trans_error_rates = []
    heur_error_rates = []
    train_policy = world.expert_policy
    for i, max_additional_seq_attempts in enumerate([20]):#[1, 4, 5, 10, 30, 50]):
        print('Adding to training dataset.')
        args.max_seq_attempts = max_additional_seq_attempts
        generate_dataset(args, world, logger, trans_dataset, heur_dataset, train_policy)
        preprocess(args, trans_dataset)
        trans_model = TransitionGNN(args, n_hidden=7)
        print('Training with %i datapoints' % len(trans_dataset))
        n_datapoints.append(len(trans_dataset))
        if args.pred_type == 'delta_state':
            loss_fn = F.mse_loss
        elif args.pred_type == 'full_state':
            loss_fn = F.binary_cross_entropy
        train(trans_dataloader, None, trans_model, n_epochs=100, loss_fn=loss_fn)
        # TODO: fix calculation
        trans_error_rate = calc_trans_error_rate(args, test_trans_dataset, trans_model)
        trans_error_rates.append(trans_error_rate)
        #print('Forward Prediction Error Rate: %f' % trans_error_rate)
        calc_successful_action_error_rate(args, test_trans_dataset, trans_model)
        if plot:
            vis_trans_errors(test_trans_dataset, trans_model)
            vis_trans_dataset_grid(trans_dataset, 'Frequency of Edges seen in Training Dataset')
            vis_trans_dataset_grid(test_trans_dataset, 'Frequency of Edges seen in Test Dataset')
            vis_trans_dataset_hist(trans_dataset, 'Tower Heights in Training Data')
        
        '''
        heur_model = HeuristicGNN()
        train(heur_dataloader, None, heur_model, n_epochs=100, loss_fn=F.mse_loss)
        heur_error_rate = calc_heur_error_rate(test_heur_dataset, heur_model)
        heur_error_rates.append(heur_error_rate)
        print('Heuristic Prediction Error Rate: %f' % heur_error_rate)
        '''

    if plot:
        # Visualize Error Rate
        fig, ax = plt.subplots()
        ax.plot(n_datapoints, trans_error_rates, '*-')
        ax.set_xlabel('Training Dataset Size (Number of Actions)')
        ax.set_ylabel('Edge Prediction Error Rate')
        
        plt.ion()
        plt.show()
        input('Enter to close plots.')
        plt.close()
    
    return n_datapoints, trans_error_rates
    
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