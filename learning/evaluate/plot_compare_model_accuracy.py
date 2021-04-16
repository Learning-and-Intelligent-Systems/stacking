import re
import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('tkagg')
import torch

from learning.evaluate.plot_model_accuracy import calc_model_accuracy
from learning.active.utils import ActiveExperimentLogger
from learning.domains.towers.active_utils import get_sequential_predictions

tower_heights = [2, 3, 4, 5, 6, 7]
min_towers_acq = 40         # number of towers in initial dataset
towers_per_acq = 10         # number of towers acquired between each trained model
separate_figs = False
                
def plot_all_model_accuracies(all_model_accuracies, all_min_accuracies, all_max_accuracies):    
    acquisition_plot_steps = len(range(0, args.max_acquisitions, args.plot_step))
    xs = np.arange(min_towers_acq, \
                    min_towers_acq+towers_per_acq*args.plot_step*acquisition_plot_steps, \
                    towers_per_acq*args.plot_step) # number of training towers

    if separate_figs:
        fig0, ax0 = plt.subplots()
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()
        axes = [ax0, ax1, ax2, ax3]
        for type in all_model_accuracies:

            for thi, (th, ax) in enumerate(zip([2, 3, 4, 5], axes)):
                ys = accuracies[thi,:]
                ax.plot(xs, ys, label=type)
                ax.set_ylim(.5, 1.)
                ax.set_ylabel('Constructability Accuracy')
                ax.set_xlabel('Training Towers')
                ax.legend()
                ax.set_title(str(th)+' Block Tower Constructability Accuracy\nAveraged Over 5 Models')
                for k in [10*10+40, 20*10+40, 30*10+40]:
                    ax.plot([k, k], [.5, 1], 'k--')

        #plt.tight_layout()
        fig0.savefig(args.output_fname+'2')
        fig1.savefig(args.output_fname+'3')
        fig2.savefig(args.output_fname+'4')
        fig3.savefig(args.output_fname+'5')
        plt.close()
    else:
        axes_map = {0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1), 4:(2,0), 5:(2,1), 6:(3,0), 7:(3,1)}
        fig, axes = plt.subplots(int(len(tower_heights)/2),2, sharex=True, sharey=True)#, figsize=(8,12))
        for type in all_model_accuracies:
            accuracies = all_model_accuracies[type]
            min_accuracies = all_min_accuracies[type]
            max_accuracies = all_max_accuracies[type]
            for thi, th in enumerate(tower_heights):
                axx, axy = axes_map[thi]
                ys = accuracies[thi,:]
                upper = min_accuracies[thi,:]
                lower = max_accuracies[thi,:]
                axes[axx, axy].plot(xs, ys, label=type)
                axes[axx, axy].fill_between(xs, upper, lower, alpha=0.2)
                axes[axx, axy].set_ylim(.5, 1.)
                if axy == 0:
                    axes[axx, axy].set_ylabel('Accuracy')
                if axx == 2:
                    axes[axx, axy].set_xlabel('Training Towers')
                #axes[axx, axy].legend()
                axes[axx, axy].set_title(str(th)+' Block')# Tower Constructability Accuracy\nAveraged Over 5 Models')
                #for k in [10*10+40, 20*10+40, 30*10+40]:
                #    axes[axx, axy].plot([k, k], [.5, 1], 'k--')
                
        # show legend
        #fig.add_subplot(111, frameon=False)
        #plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        #plt.xlabel("common X")
        #plt.ylabel("common Y")
        lgd = axes[axx, axy].legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, bbox_transform=fig.transFigure)
        fig.subplots_adjust(left=0.13, bottom=0.11, right=0.9, top=0.88, wspace=.1, hspace=0.35)
        title = fig.suptitle('Model Accuracy for Different Tower Sizes')
        #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(args.output_fname+'.pdf', bbox_extra_artists=(lgd,title), bbox_inches='tight')
        #plt.show()
        #plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-acquisitions',
                        type=int, 
                        help='evaluate from 0 to this acquisition step (use either this or --acquisition-step)')
    parser.add_argument('--plot-step',
                        type=int,
                        default=10)
    parser.add_argument('--single-acquisition-step',
                        type=int,
                        help='evaluate only this acquisition step(use either this or --max-acquisitions)')
    parser.add_argument('--test-set-fname',
                        type=str,
                        required=True,
                        help='evaluate only this acquisition step(use either this or --max-acquisitions)')                        
    parser.add_argument('--output-fname',
                        type=str,
                        required=True,
                        help='evaluate only this acquisition step(use either this or --max-acquisitions)')                        
    parser.add_argument('--debug',
                        action='store_true',
                        help='set to run in debug mode')
    
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    with open(args.test_set_fname, 'rb') as f:
        dataset = pickle.load(f)
        
    # model architecture comparison
    exp_paths = {'sequential TGN': 
                    ['paper_results_02232021/subtower-sequential-fcgn-0-20210223-223622',
                    'paper_results_02232021/subtower-sequential-fcgn-1-20210227-085527',
                    'paper_results_02232021/subtower-sequential-fcgn-2-20210227-043159'],
                'sequential FCGN': 
                    ['paper_results_02232021/subtower-sequential-fcgn-fc-0-20210225-235513',
                    'paper_results_02232021/subtower-sequential-fcgn-fc-1-20210227-003015',
                    'paper_results_02232021/subtower-sequential-fcgn-fc-2-20210227-185408'],
                'sequential LSTM': 
                    ['paper_results_02232021/subtower-sequential-lstm-0-20210225-235514',
                    'paper_results_02232021/subtower-sequential-lstm-1-20210227-033252',
                    'paper_results_02232021/subtower-sequential-lstm-2-20210227-185647']}
    
    all_accuracies = {}
    min_accuracies = {}
    max_accuracies = {}
    for set_type, exp_path_set in exp_paths.items():
        num_models = len(exp_path_set)
        n_xs = len(list(range(0, args.max_acquisitions, args.plot_step)))
        exp_path_set_accuracies = np.zeros((len(tower_heights), num_models, n_xs))
        for i, full_exp_path in enumerate(exp_path_set):
            logger = ActiveExperimentLogger(full_exp_path)
            model_accuracies = calc_model_accuracy(logger, dataset, args, full_exp_path, save_local_fig=False)
            #print(model_accuracies)
            for ti, (tower_height, height_accuracies) in enumerate(model_accuracies.items()):
                if tower_height in dataset:
                    exp_path_set_accuracies[ti,i,:] = model_accuracies[tower_height]
        exp_path_set_avg_accuracies = exp_path_set_accuracies.mean(axis=1)
        exp_path_set_min_accuracies = exp_path_set_accuracies.min(axis=1)
        exp_path_set_max_accuracies = exp_path_set_accuracies.max(axis=1)
        #print(exp_path_set_avg_accuracies)
        all_accuracies[set_type] = exp_path_set_avg_accuracies
        min_accuracies[set_type] = exp_path_set_min_accuracies
        max_accuracies[set_type] = exp_path_set_max_accuracies
        
    plot_all_model_accuracies(all_accuracies, min_accuracies, max_accuracies)