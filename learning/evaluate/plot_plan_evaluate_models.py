import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

from learning.active.utils import ActiveExperimentLogger
from learning.evaluate.active_evaluate_towers import plot_tallest_tower_regret

# CONSTANTS
min_towers_acq = 40         # number of towers in initial dataset
towers_per_acq = 10         # number of towers acquired between each trained model
acquisition_step_size = 10  # steps between models evaluated

def plot_planner_performance(logger, fname, args, y_axis, title):
    fig, ax = plt.subplots()

    with open(logger.get_figure_path(fname), 'rb') as handle:
        metric = pickle.load(handle)
    rs = metric[str(args.tower_size)+'block']
    n_towers = len(rs[0])
    
    acquisition_plot_steps = len(range(0, args.max_acquisitions, acquisition_step_size))
    txs = list(range(0, n_towers*acquisition_plot_steps, n_towers)) # indices into lists
    xs = np.arange(min_towers_acq, \
                    min_towers_acq+towers_per_acq*acquisition_step_size*acquisition_plot_steps, \
                    towers_per_acq*acquisition_step_size) # number of training towers

    median, lower25, upper75 = [], [], []
    for tx in range(len(rs)):
        median.append(np.median(rs[tx]))
        lower25.append(np.quantile(rs[tx], 0.25))
        upper75.append(np.quantile(rs[tx], 0.75))
         
    ax.plot(xs, median)
    ax.fill_between(xs, lower25, upper75, alpha=0.2)
     
    ax.set_ylabel(y_axis)
    ax.set_xlabel('Number of Training Towers')
    ax.set_title(title)
    #ax.legend()
    plt_fname = fname[:-4]+'.png'
    plt.savefig(logger.get_figure_path(plt_fname))

    plt.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', 
                        choices=['tallest', 'overhang', 'min-contact', 'deconstruct'], 
                        default='tallest',
                        help='planning problem/task to plan for')
    parser.add_argument('--exp-path', 
                        type=str, 
                        required=True)
    parser.add_argument('--debug',
                        action='store_true',
                        help='set to run in debug mode')
    parser.add_argument('--tower-sizes',
                        default=[5],
                        nargs = '+',
                        help='number of blocks in goal tower')
    parser.add_argument('--discrete',
                        action='store_true',
                        help='use if you want to ONLY search the space of block orderings and orientations')
    parser.add_argument('--plot-type',
                        default='regret',
                        choices=['regret', 'reward'])
    parser.add_argument('--max-acquisitions',
                        type=int, 
                        help='evaluate from 0 to this acquisition step (use either this or --acquisition-step)')
    
    args = parser.parse_args()
    
    args.tower_sizes = [int(ts) for ts in args.tower_sizes]
    
    if args.debug:
        import pdb; pdb.set_trace()
 
    logger = ActiveExperimentLogger(args.exp_path)
    fname = 'discrete_' if args.discrete else ''
    
    ts_str = ''.join([str(ts) for ts in args.tower_sizes])
    if args.problem == 'tallest':
        fname += 'random_planner_tallest_tower_'+ts_str+'_block_towers'
        title = 'Tallest Tower'
    elif args.problem == 'overhang':
        fname += 'random_planner_max_overhang_'+ts_str+'_block_towers'
        title = 'Maximum Overhang'
    elif args.problem == 'min-contact':
        fname += 'random_planner_min_contact_'+ts_str+'_block_towers'
        title = 'Minimum Contact'
        
    if args.plot_type == 'regret':
        fname += '_regrets.pkl'
        y_axis = 'Normalized Median Regret'
    elif args.plot_type == 'reward':
        fname += '_rewards.pkl'
        y_axis = 'Normalized Median Reward'
        
    plot_planner_performance(logger, fname, args, y_axis, title)