import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

from learning.active.utils import ActiveExperimentLogger
from learning.evaluate.plot_plan_evaluate_models import plot_planner_performance

def plot_all_task_performance(xs, plot_task_data, args, task):
    fig, axes = plt.subplots(len(args.tower_sizes), figsize=(7, 15))
    for mi, (method, method_plot_data) in enumerate(plot_task_data.items()):
        for ai, (tower_size, plot_data) in enumerate(method_plot_data.items()):
            axes[ai].plot(xs, plot_data['median'], label=method)
            axes[ai].fill_between(xs, plot_data['lower25'], plot_data['upper75'], alpha=0.2)
            axes[ai].set_ylim(0, 1)
            axes[ai].set_ylabel('Regret')
            axes[ai].set_xlabel('Number of Training Towers')
            axes[ai].legend()
            
    fig.suptitle(task)
    plot_dir = 'learning/experiments/logs/paper_plots/compare_methods/'
    if not os.path.exists(plot_dir): os.makedirs(plot_dir)
    plt.savefig('learning/experiments/logs/paper_plots/compare_methods/'+task+'.png')
    plt.close()
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-path-roots', 
                        nargs='+',
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
    
    args = parser.parse_args()
    
    args.tower_sizes = [int(ts) for ts in args.tower_sizes]
    
    if args.debug:
        import pdb; pdb.set_trace()
 
    tallest_tower_plot_data = {}
    min_contact_plot_data = {}
    max_overhang_plot_data = {}
    all_plot_data = [tallest_tower_plot_data, min_contact_plot_data, max_overhang_plot_data]
    y_axis = 'Regret' # TODO: detect from file name?
    for exp_path_root in args.exp_path_roots:
        # find exp_paths with the given root in paper_results/
        results_path = 'paper_results'
        exp_path_roots = [exp_path_root+'-'+str(r) for r in [0, 1, 2, 3, 4]]
        all_paper_results = os.listdir(results_path)
        exp_paths = []
        for result in all_paper_results:
            for exp_path_root in exp_path_roots:
                if exp_path_root in result:
                    exp_paths.append(result)
        print(exp_paths)
        
        loggers = []
        for exp_path in exp_paths:
            loggers.append(ActiveExperimentLogger(os.path.join(results_path, exp_path)))
            
        
        label = exp_path_root
        fnames = ['random_planner_tallest2345_block_towers_regrets.pkl', \
                    'random_planner_min_contact_2345_block_towers_regrets.pkl', \
                    'random_planner_max_overhang_2345_block_towers_regrets.pkl']
        for fname, task_plot_data in zip(fnames, all_plot_data):
            xs, plot_data = plot_planner_performance(loggers, args, y_axis, label, fname)
            task_plot_data[label] = plot_data
            
    tasks = ['Tallest Tower', 'Minimum Contact', 'Maximum Overhang']
    for task, task_plot_data in zip(tasks, all_plot_data):
        plot_all_task_performance(xs, task_plot_data, args, task)