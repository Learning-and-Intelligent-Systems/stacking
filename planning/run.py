import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

from learning.active.utils import ActiveExperimentLogger
from block_utils import Object
#from planning.plan import plan as sequential_planner
from planning.plan import plan_mcts as sequential_planner
from planning.problems import Tallest
from learning.evaluate.active_evaluate_towers import tallest_tower_regret_evaluation as total_planner
from learning.evaluate.active_evaluate_towers import plot_tallest_tower_regret

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method',
                        choices=['sequential', 'total', 'both'],
                        default='sequential',
                        help='use sequention method (tree building) or totally \
                                random method (towers are generated randomly) \
                                to plan, or compare both methods')
    parser.add_argument('--problem', 
                        choices=['tallest', 'overhang', 'deconstruct'], 
                        default='tallest',
                        help='planning problem/task to plan for')
    parser.add_argument('--block-set-fname', 
                        type=str, 
                        default='',
                        help='path to the block set file. if not set, random 5 blocks generated.')
    parser.add_argument('--exp-path', 
                        type=str, 
                        required=True)
    parser.add_argument('--max-acquisitions',
                        type=int, 
                        required=True)
    parser.add_argument('--n-towers',
                        default = 50,
                        type=int,
                        help = 'number of tall towers to find for each acquisition step')
    parser.add_argument('--n-blocks',
                        default = 5,
                        type = int,
                        help='number of blocks in random block set (if block-set-fname is not set)')
    parser.add_argument('--timeout',
                        type=int,
                        default=1000,
                        help='max number of iterations to run planner for')
    parser.add_argument('--debug',
                        action='store_true',
                        help='set to run in debug mode')
    parser.add_argument('--discrete',
                        action='store_true',
                        help='use if you want to ONLY search the space of block orderings and orientations')
    args = parser.parse_args()
    
    if args.debug:
        import pdb; pdb.set_trace()
 
    if args.block_set_fname is not '':
        with open(args.block_set_fname, 'rb') as f:
            block_set = pickle.load(f)
    else:
        block_set = [Object.random(f'obj_{ix}') for ix in range(args.n_blocks)]

    logger = ActiveExperimentLogger(args.exp_path)
    pre = 'discrete_' if args.discrete else ''
    
    ## RUN SEQUENTIAL PLANNER
    if args.method == 'sequential' or args.method == 'both':
        tower_keys = ['2block', '3block', '4block', '5block']
        tower_sizes = [2, 3, 4, 5]
        max_height = 5
        
        # Store regret for towers of each size.
        regrets = {k: [] for k in tower_keys}

        for tx in range(0, args.max_acquisitions, 10):
            print('Acquisition step:', tx)
            ensemble = logger.get_ensemble(tx)

            problem = Tallest(max_height)
            tx_regrets = {k: [] for k in tower_keys}
            for t in range(0, args.n_towers):
                print('Tower number', t+1, '/', args.n_towers)
                # generate new block set for each tower search
                search_tree = sequential_planner(args.timeout, block_set, problem, ensemble, discrete=args.discrete)
                for i, (k, size) in enumerate(zip(tower_keys, tower_sizes)):
                    print('Finding best tower size: ', size)
                    exp_best_node_id = search_tree.get_exp_best_node(size)
                    if exp_best_node_id is not None:
                        best_tower = search_tree.nodes[exp_best_node_id]['tower']
                        reward = search_tree.nodes[exp_best_node_id]['tower_height']
                        gt_best_node_id = search_tree.get_ground_truth_best_node(size)
                        gt_reward = search_tree.nodes[gt_best_node_id]['ground_truth']

                        if not problem.tp.tower_is_constructable(best_tower):
                            reward = 0
                        regret = (gt_reward - reward)/gt_reward
                    else:
                        print('None found.')
                        regret = 1
                    tx_regrets[k].append(regret)
            for k in tower_keys:
                regrets[k].append(tx_regrets[k])
                
            with open(logger.get_figure_path(pre+'sequential_planner_tallest_tower_regret.pkl'), 'wb') as handle:
                pickle.dump(regrets, handle)

    
    ## RUN RANDOM PLANNER
    if args.method == 'total' or args.method == 'both':
        total_planner(logger, args.max_acquisitions, pre+'total_planner_tallest_tower_regret.pkl', \
            args.n_towers, block_set, discrete=args.discrete)

    ## PLOT RESULTS
    if args.method in ['sequential', 'both']:
        plot_tallest_tower_regret(logger, pre+'sequential_planner_tallest_tower_regret.pkl')
    if args.method in ['total', 'both']:
        plot_tallest_tower_regret(logger, pre+'total_planner_tallest_tower_regret.pkl')