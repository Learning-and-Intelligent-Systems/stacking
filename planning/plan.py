from collections import namedtuple
import argparse
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from planning.tree import Tree
from planning.problems import Tallest, Overhang, Deconstruct
from block_utils import Object
from learning.active.utils import ActiveExperimentLogger

def plan(timeout, blocks, problem, model):
    tree = Tree(blocks)
    for t in range(timeout):
        parent_node_id = tree.get_exp_best_node_expand()
        #print(t, len(tree.nodes[parent_node_id]['tower']), tree.nodes[parent_node_id]['value'])
        sys.stdout.write("Search progress: %i   \r" % (t) )
        sys.stdout.flush()
        new_nodes = problem.sample_actions(tree.nodes[parent_node_id], model)
        for node in new_nodes:
            tree.expand(parent_node_id, node)
    return tree

def plan_mcts(timeout, blocks, problem, model, c=np.sqrt(2), discrete=True):
    tree = Tree(blocks)
    tallest_tower = [0]
    highest_exp_height = [0]
    highest_value = [0]
    tower_stats = np.zeros((5,timeout))
    
    for t in range(timeout):
        tower_stats[:,t] = tower_stats[:,t-1]
        sys.stdout.write("Search progress: %i   \r" % (t) )
        sys.stdout.flush()
        parent_node_id = tree.traverse(c)
        
        new_nodes = problem.sample_actions(tree.nodes[parent_node_id], model, discrete=discrete)
        tallest_tower_t = tallest_tower[-1]
        highest_exp_height_t = highest_exp_height[-1]
        highest_value_t = highest_value[-1]
        for new_node in new_nodes:
            #print(t, len(new_node['tower']), new_node['exp_reward'])
            new_node_id = tree.expand(parent_node_id, new_node)
            rollout_value = tree.rollout(new_node_id, problem, model)
            tree.backpropagate(new_node_id, rollout_value)
            
            tower_height = len(new_node['tower'])
            #print(tower_height)
            index = int(tower_height)
            tower_stats[index-1,t] += 1
            if len(new_node['tower'])>tallest_tower_t:
                tallest_tower_t = len(new_node['tower'])
                
            if new_node['exp_reward'] > highest_exp_height_t:
                highest_exp_height_t = new_node['exp_reward']
                
            if new_node['value'] > highest_value_t:
                highest_value_t = new_node['value']
        tallest_tower.append(tallest_tower_t)
        highest_exp_height.append(highest_exp_height_t)
        highest_value.append(highest_value_t)

    return tree, tallest_tower, highest_exp_height, highest_value, tower_stats
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--timeout',
                        type=int,
                        default=1000,
                        help='max number of iterations to run planner for')
    parser.add_argument('--debug',
                        action='store_true',
                        help='set to run in debug mode')
    parser.add_argument('--max-height',
                        type=int,
                        default=5,
                        help='number of blocks in desired tower')
    args = parser.parse_args()
    
    if args.debug:
        import pdb; pdb.set_trace()
    
    n_blocks = 5
    tx = 99
    
    if args.block_set_fname is not '':
        with open(args.block_set_fname, 'rb') as f:
            block_set = pickle.load(f)
    else:
        block_set = [Object.random(f'obj_{ix}') for ix in range(n_blocks)]
        
    if args.problem == 'tallest':
        problem = Tallest(args.max_height)
    elif args.problem == 'overhang':
        problem = Overhang()
    elif args.problem == 'deconstruct':
        problem = Deconstruct()
        
    logger = ActiveExperimentLogger(args.exp_path)
    ensemble = logger.get_ensemble(tx)
    
    c_vals = [0, 0.5, 1, np.sqrt(2)]#10 ** np.linspace(0,10)
    
    for c in c_vals:
        runs = 1
        all_tallest_towers = np.zeros((args.timeout+1, runs))
        all_highest_exp_heights = np.zeros((args.timeout+1, runs))
        all_highest_values = np.zeros((args.timeout+1, runs))
        for run in range(runs):
            tallest_tower, highest_exp_height, highest_value, tree, tower_stats = \
                plan_mcts(args.timeout, block_set, problem, ensemble, c=c)
                
            #all_tallest_towers[:,run] = tallest_tower
            #all_highest_exp_heights[:,run] = highest_exp_height
            #all_highest_values[:,run] = highest_value
            
            plt.figure()
            xs = list(range(tower_stats.shape[1]))
            keys = ['2block', '3block', '4block', '5block']
            plt.bar(xs, tower_stats[0,:], label='1block')
            for i, key in enumerate(keys):
                plt.bar(xs, tower_stats[i+1,:], bottom=np.sum(tower_stats[:i+1,:], axis=0), label=key)
            plt.title('c= '+str(c))
            plt.legend()
            timestamp = datetime.now().strftime("%d-%m-%H-%M-%S")
            plt.savefig('mcts_test_hist_'+str(timestamp))
            #plt.show()
            '''
            median_tt = np.median(all_tallest_towers, axis=1)
            median_hev = np.median(all_highest_exp_heights, axis=1)
            median_hv = np.median(all_highest_values, axis=1)
            
            q25_tt = np.quantile(all_tallest_towers, 0.25, axis=1)
            q75_tt = np.quantile(all_tallest_towers, 0.75, axis=1)
            
            q25_hev = np.quantile(all_highest_exp_heights, 0.25, axis=1)
            q75_hev = np.quantile(all_highest_exp_heights, 0.75, axis=1)
            
            q25_hv = np.quantile(all_highest_values, 0.25, axis=1)
            q75_hv = np.quantile(all_highest_values, 0.75, axis=1)
            '''
            fig, ax = plt.subplots(3)
            
            xs = list(range(len(tallest_tower)))
            ax[0].plot(xs, tallest_tower, label='tallest tower')
            #ax[0].fill_between(xs, q25_tt, q75_tt, alpha=0.2)
            
            ax[1].plot(xs, highest_exp_height, label='highest expected height')
            #ax[1].fill_between(xs, q25_hev, q75_hev, alpha=0.2)
            
            ax[2].plot(xs, highest_value, label='highest node value')
            #ax[2].fill_between(xs, q25_hv, q75_hv, alpha=0.2)
            
            ax[0].legend()
            ax[1].legend()
            ax[2].legend()
            
            ax[0].set_ylim(0.0, 5.1)
            ax[1].set_ylim(0.0, 0.6)
            ax[2].set_ylim(0.0, 0.6)
            
            ax[0].set_title('c='+str(c))
            
            timestamp = datetime.now().strftime("%d-%m-%H-%M-%S")
            plt.savefig('mcts_test_'+str(timestamp))
            