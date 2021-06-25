import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from learning.active.utils import ActiveExperimentLogger
from learning.evaluate.plot_plan_evaluate_models import plot_planner_performance

def plot_all_task_performance(xs, plot_task_data, args, task, y_axis, color_map):
    plot_height = 4*len(args.tower_sizes)
    fig, axes = plt.subplots(len(args.tower_sizes), figsize=(7, plot_height))
    for mi, (method, method_plot_data) in enumerate(plot_task_data.items()):
        #print(method)
        if len(method_plot_data) == 1: # just plotting a single tower height
            key = list(method_plot_data.keys())[0]
            if len(method_plot_data[key]['mid']) == 1: # single value to plot along entire x axis (eg. random strategy)
                axes.plot(xs, method_plot_data[key]['mid']*len(xs), label=method, color=color_map[method],)
                axes.fill_between(xs, method_plot_data[key]['lower']*len(xs), method_plot_data[key]['upper'], color=color_map[method], alpha=0.2)
                axes.plot(xs, method_plot_data[key]['lower']*len(xs), color=color_map[method], alpha=0.1)
                axes.plot(xs, method_plot_data[key]['upper']*len(xs), color=color_map[method], alpha=0.1)
                #axes.set_ylim(0, 1)
                axes.set_ylabel(y_axis)
                axes.set_xlabel('Number of Training Towers')
                axes.legend(loc='upper right')
            else:
                axes.plot(xs, method_plot_data[key]['mid'], label=method, color=color_map[method],)
                axes.fill_between(xs, method_plot_data[key]['lower'], method_plot_data[key]['upper'], color=color_map[method], alpha=0.2)
                axes.plot(xs, method_plot_data[key]['lower'], color=color_map[method], alpha=0.1)
                axes.plot(xs, method_plot_data[key]['upper'], color=color_map[method], alpha=0.1)
                #axes.set_ylim(0, 1)
                axes.set_ylabel(y_axis)
                axes.set_xlabel('Number of Training Towers')
                axes.legend(loc='upper right')
        else: # plotting mumtiple tower heights
            for ai, (tower_size, plot_data) in enumerate(method_plot_data.items()):
                axes[ai].plot(xs, plot_data['mid'], label=method)
                axes[ai].fill_between(xs, plot_data['lower'], plot_data['upper'], alpha=0.2)
                #axes[ai].set_ylim(0, 1)
                axes[ai].set_ylabel(y_axis)
                axes[ai].set_xlabel('Number of Training Towers')
                axes[ai].legend()
            
    fig.suptitle('%d Block %s Towers' % (args.tower_size, task))
    plot_dir = 'figures'
    if not os.path.exists('figures'):
        os.makedirs('figures')
    timestamp = datetime.now().strftime("%d-%m-%H-%M-%S")
    plt.savefig('%s_%s.pdf' % (os.path.join('figures', task), timestamp))
    plt.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',
                        action='store_true',
                        help='set to run in debug mode')
    parser.add_argument('--tower-size',
                        default=5,
                        type=int,
                        help='number of blocks in goal tower')
    parser.add_argument('--problems',
                        default=['tallest', 'min_contact', 'max_overhang'],
                        nargs='+')
    parser.add_argument('--max-acquisitions',
                        type=int, 
                        help='evaluate from 0 to this acquisition step (use either this or --acquisition-step)')
    
    args = parser.parse_args()
    args.tower_sizes = [args.tower_size] # plot_planner_performance needs this in args
    if args.debug:
        import pdb; pdb.set_trace()
 
    tallest_tower_plot_data = {}
    min_contact_plot_data = {}
    max_overhang_plot_data = {}
    all_plot_data = [tallest_tower_plot_data, min_contact_plot_data, max_overhang_plot_data]
    y_axis = 'Normalized Regret'
    # Options:
    #   'Normalized Regret'
    #   'Success Rate'
    #   'Average Successful Reward'
    #   'Average Non Zero Reward'
    #   'Non Zero Reward Rate'
    #   'Stable Towers Regret'
    
    
    all_exp_paths = {'incremental':
                        ['rss_camera_ready/bald-sequential-fcgn-0-20210604-160506',
                            'rss_camera_ready/bald-sequential-fcgn-1-20210604-190943',
                            'rss_camera_ready/bald-sequential-fcgn-2-20210604-193703',
                            'rss_camera_ready/bald-sequential-fcgn-3-20210615-231115'],
                'greedy':        
                        ['rss_camera_ready/subtower-greedy-sequential-fcgn-0-20210604-160500',
                            'rss_camera_ready/subtower-greedy-sequential-fcgn-1-20210604-190834',
                            'rss_camera_ready/subtower-greedy-sequential-fcgn-2-20210604-193549',
                            'rss_camera_ready/subtower-greedy-sequential-fcgn-3-20210615-145846'],
                'random-comp':        
                        ['rss_camera_ready/random-random-fcgn-0-20210604-193209',
                            'rss_camera_ready/random-random-fcgn-1-20210604-195044',
                            'rss_camera_ready/random-random-fcgn-2-20210604-195352',
                            'rss_camera_ready/random-random-fcgn-3-20210615-231130'],
                'complete':
                        ['rss_camera_ready/bald-random-fcgn-0-20210604-160822',
                            'rss_camera_ready/bald-random-fcgn-1-20210604-201218',
                            'rss_camera_ready/bald-random-fcgn-2-20210604-193847',
                            'rss_camera_ready/bald-random-fcgn-3-20210615-145732'],
                'random-ss':
                        ['rss_camera_ready/subtower-random-fcgn-0-20210620-121050',
                            'rss_camera_ready/subtower-random-fcgn-1-20210620-121152',
                            'rss_camera_ready/subtower-random-fcgn-2-20210622-042118',
                            'rss_camera_ready/subtower-random-fcgn-3-20210622-042847'],
                'sequential':
                        ['rss_camera_ready/subtower-sequential-fcgn-0-20210604-160455',
                            'rss_camera_ready/subtower-sequential-fcgn-1-20210604-190759',
                            'rss_camera_ready/subtower-sequential-fcgn-2-20210604-193420',
                            'rss_camera_ready/subtower-sequential-fcgn-3-20210616-091302']}
    all_color_map = {'incremental': '#ff7f0e', 
                    'greedy': '#1f77b4', 
                    'random-comp': '#800080',
                    'complete': '#2ca02c',
                    'random-ss': '#000000',
                    'sequential': '#d62728'}

    #### set this
    #methods_to_plot = ['sequential', 'incremental', 'greedy', 'complete']
    #methods_to_plot = ['sequential', 'incremental', 'greedy', 'random-ss']
    methods_to_plot = ['complete', 'random-comp']
    ####
    
    exp_paths = {method: all_exp_paths[method] for method in methods_to_plot}
    color_map = {method: all_color_map[method] for method in methods_to_plot}

    all_problems = ['tallest', 'min_contact', 'max_overhang']
    for label, same_exp_paths in exp_paths.items():
        loggers = []
        for exp_path in same_exp_paths:
            loggers.append(ActiveExperimentLogger(exp_path))

        for problem, task_plot_data in zip(all_problems, all_plot_data):
            if problem in args.problems:
                regret_fname = 'random_planner_%s_%d_block_towers_regrets.pkl' % (problem, args.tower_size)
                reward_fname = 'random_planner_%s_%d_block_towers_rewards.pkl' % (problem, args.tower_size)
                xs, plot_data = plot_planner_performance(loggers, args, y_axis, label, regret_fname, reward_fname)
                task_plot_data[label] = plot_data

    tasks = ['Tallest Tower', 'Maximum Unsupported Area', 'Longest Overhang']
    for task, task_arg, task_plot_data in zip(tasks, all_problems, all_plot_data):
        if task_arg in args.problems:
            plot_all_task_performance(xs, task_plot_data, args, task, y_axis, color_map)