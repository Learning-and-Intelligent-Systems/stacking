import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

from learning.active.utils import ActiveExperimentLogger
from learning.evaluate.active_evaluate_towers import plot_regret

# CONSTANTS
min_towers_acq = 40         # number of towers in initial dataset
towers_per_acq = 10         # number of towers acquired between each trained model
acquisition_step_size = 10  # steps between models evaluated

def plot_planner_performance(loggers, args, y_axis, method, fname):
    all_rs = {str(ts)+'block' : [] for ts in args.tower_sizes}
    for logger in loggers:
        for tower_size in args.tower_sizes:
            with open(logger.get_figure_path(fname), 'rb') as handle:
                metric = pickle.load(handle)
            try:
                tower_size_key = str(tower_size)+'block'
                rs = metric[tower_size_key]
            except:
                print('tower sizes '+str(tower_size)+' are not in file '+fname)
            all_rs[tower_size_key].append(rs)
    
    plot_data = {}
    for tsk in all_rs:
        n_towers = len(all_rs[tsk][0][0])
        acquisition_plot_steps = len(range(0, args.max_acquisitions, acquisition_step_size))
        txs = list(range(0, n_towers*acquisition_plot_steps, n_towers)) # indices into lists

        median, lower25, upper75 = [], [], []
        for tx in range(len(all_rs[tsk][0])):
            tx_rs = []
            for rs in all_rs[tsk]:
                tx_rs.append(rs[tx])
            median.append(np.median(tx_rs))
            lower25.append(np.quantile(tx_rs, 0.25))
            upper75.append(np.quantile(tx_rs, 0.75))
             
        plot_data[tsk] = {}
        plot_data[tsk]['median'] = median
        plot_data[tsk]['lower25'] = lower25
        plot_data[tsk]['upper75'] = upper75

    xs = np.arange(min_towers_acq, \
                    min_towers_acq+towers_per_acq*acquisition_step_size*acquisition_plot_steps, \
                    towers_per_acq*acquisition_step_size) # number of training towers

    fig_height = 4.5*len(args.tower_sizes)
    fig, axes = plt.subplots(len(args.tower_sizes), figsize=(7, fig_height))
    if len(args.tower_sizes) == 1:
        key = list(plot_data.keys())[0]
        axes.plot(xs, plot_data[key]['median'], label=tsk)
        axes.fill_between(xs, plot_data[key]['lower25'], plot_data[key]['upper75'], alpha=0.2)
        axes.set_ylim(0, 1)
        axes.set_ylabel(y_axis)
        axes.set_xlabel('Number of Training Towers')
        axes.legend()
    else:
        for i, (tsk, plot_data_tsk) in enumerate(plot_data.items()):
            axes[i].plot(xs, plot_data_tsk['median'], label=tsk)
            axes[i].fill_between(xs, plot_data_tsk['lower25'], plot_data_tsk['upper75'], alpha=0.2)
            axes[i].set_ylim(0, 1)
            axes[i].set_ylabel(y_axis)
            axes[i].set_xlabel('Number of Training Towers')
            axes[i].legend()
        
    fig.suptitle(method)
    
    if len(loggers) > 1:
        plot_dir = 'learning/experiments/logs/paper_plots/combine_models/'+method
        if not os.path.exists(plot_dir): os.makedirs(plot_dir)
        fig_path = os.path.join(plot_dir, fname[:-4]+'.png')
        plt.savefig(fig_path)
        print('Saved to %s.' % fig_path)
    else:
        plt.savefig(loggers[0].get_figure_path(fname[:-4]+'.png'))
    plt.close()
    
    return xs, plot_data
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-path-root',
                        type=str,
                        required=True)
    parser.add_argument('--exp-path-prefix', 
                        type=str,
                        required=True)
    parser.add_argument('--debug',
                        action='store_true',
                        help='set to run in debug mode')
    parser.add_argument('--tower-sizes',
                        default=[2, 3, 4, 5],
                        nargs = '+',
                        help='number of blocks in goal tower')
    parser.add_argument('--max-acquisitions',
                        type=int, 
                        help='evaluate from 0 to this acquisition step (use either this or --acquisition-step)')
    parser.add_argument('--runs',
                        nargs='+',
                        default=[0, 1, 2, 3, 4])
    parser.add_argument('--problems',
                        default=['tallest', 'min_contact', 'max_overhang'],
                        nargs='+')
    args = parser.parse_args()
    
    args.tower_sizes = [int(ts) for ts in args.tower_sizes]
    
    if args.debug:
        import pdb; pdb.set_trace()
 
    # find exp_paths with the given prefix
    exp_path_roots = ['%s-%d' % (args.exp_path_prefix, r) for r in args.runs]
    all_results = os.listdir(args.exp_path_root)
    relevant_exp_paths = []
    for result in all_results:
        for exp_path_root in exp_path_roots:
            if exp_path_root in result:
                relevant_exp_paths.append(os.path.join(args.exp_path_root, result))
    print(relevant_exp_paths)
    
    loggers = []
    for exp_path in relevant_exp_paths:
        loggers.append(ActiveExperimentLogger(exp_path))
        
    y_axis = 'Regret' # TODO: detect from file name?
    label = args.exp_path_prefix
    fnames = []
    for tower_size in args.tower_sizes:
        for problem in args.problems:
            fnames += ['random_planner_%s_%d_block_towers_regrets.pkl' % (problem, tower_size)]
    for fname in fnames:
        plot_planner_performance(loggers, args, y_axis, label, fname)