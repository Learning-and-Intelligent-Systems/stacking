import argparse
import torch
from torch.utils.data import DataLoader
from copy import copy
from torch.nn import functional as F
import matplotlib.pyplot as plt
import matplotlib

from learning.active.train import train
from learning.active.utils import GoalConditionedExperimentLogger
from learning.models.goal_conditioned import TransitionGNN, HeuristicGNN
from learning.domains.abc_blocks.abc_blocks_data import ABCBlocksTransDataset, ABCBlocksHeurDataset

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
                        choices=['delta_state', 'full_state', 'class'],
                        default='delta_state')
    parser.add_argument('--exp-name',
                        type=str,
                        required=True,
                        help='path to save dataset to')
    #parser.add_argument('--dataset-exp-path',
    #                    type=str,
    #                    required=True,
    #                    help='path to folder with training dataset')
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

    dataset_paths = ['learning/experiments/logs/datasets/random-actions-100-20210818-224843',
                        'learning/experiments/logs/datasets/random-actions-100-20210818-224843_1',
                        'learning/experiments/logs/datasets/random-actions-100-20210818-224843_2',
                        'learning/experiments/logs/datasets/random-actions-100-20210818-224843_3',
                        'learning/experiments/logs/datasets/random-actions-100-20210818-224843_4',
                        'learning/experiments/logs/datasets/random-actions-100-20210818-224843_5',
                        'learning/experiments/logs/datasets/random-actions-100-20210818-224843_6',
                        'learning/experiments/logs/datasets/random-actions-100-20210818-224843_7',
                        'learning/experiments/logs/datasets/random-actions-100-20210818-224843_8',
                        'learning/experiments/logs/datasets/random-actions-100-20210818-224844',
                        'learning/experiments/logs/datasets/random-actions-100-20210818-224844_1',
                        'learning/experiments/logs/datasets/random-actions-100-20210818-224844_2',
                        'learning/experiments/logs/datasets/random-actions-100-20210818-224844_3',
                        'learning/experiments/logs/datasets/random-actions-100-20210818-224844_4',
                        'learning/experiments/logs/datasets/random-actions-100-20210818-224844_5',
                        'learning/experiments/logs/datasets/random-actions-100-20210818-224844_6',
                        'learning/experiments/logs/datasets/random-actions-100-20210818-224844_7',
                        'learning/experiments/logs/datasets/random-actions-100-20210818-224844_8',
                        'learning/experiments/logs/datasets/random-actions-100-20210818-224845',
                        'learning/experiments/logs/datasets/random-actions-100-20210818-224845_1']

    for dataset_path in dataset_paths:
        args.dataset_exp_path = dataset_path
        #try:
        n_of_in=1
        n_ef_in=1
        n_af_in=2

        dataset_logger = GoalConditionedExperimentLogger(args.dataset_exp_path)
        trans_dataset = dataset_logger.load_trans_dataset()
        trans_dataset.set_pred_type(args.pred_type)
        #heur_dataset = dataset_logger.load_heur_dataset()

        # add num_blocks to model args
        dataset_args = dataset_logger.load_args()
        args.num_blocks = dataset_args.num_blocks
        #args.policy = dataset_args.policy

        model_logger = GoalConditionedExperimentLogger.setup_experiment_directory(args, 'models')

        print('Training datasets %s.' % dataset_logger.exp_path)
        trans_dataloader = DataLoader(trans_dataset, batch_size=args.batch_size, shuffle=True)
        trans_model = TransitionGNN(n_of_in=n_of_in,
                                    n_ef_in=n_ef_in,
                                    n_af_in=n_af_in,
                                    n_hidden=args.n_hidden,
                                    pred_type=args.pred_type)
        #heur_dataloader = DataLoader(heur_dataset, batch_size=args.batch_size, shuffle=True)
        #heur_model = HeuristicGNN(n_of_in=n_of_in,
        #                            n_ef_in=n_ef_in,
        #                            n_hidden=args.n_hidden)
        if args.pred_type == 'delta_state':
            loss_fn = F.mse_loss
        elif args.pred_type == 'full_state' or args.pred_type == 'class':
            loss_fn = F.binary_cross_entropy
        print('Training transition model with %i datapoints.' % len(trans_dataset))
        train(trans_dataloader, None, trans_model, n_epochs=args.n_epochs, loss_fn=loss_fn)

        #print('Training heuristic model with %i datapoints.' % len(heur_dataset))
        #train(heur_dataloader, None, heur_model, n_epochs=args.n_epochs, loss_fn = F.mse_loss)

        model_logger.save_trans_model(trans_model)
        #model_logger.save_heur_model(heur_model)
        print('Saved models to %s.' % model_logger.exp_path)
        #except:
        #    import pdb; pdb.post_mortem()
