import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from learning.active.utils import ActiveExperimentLogger

## PARAMS TO SET BEFORE RUNNING PLOTTING CODE
max_height = 3              # max num blocks in a tower in search tree
tower_key = '3block'        # goal number of blocks in tower
timeouts = [1000]              # timeouts used in .pkl files to be plotted (sequential planning)
n_samples = [50]            # number of sampled used in .pkl files to be plotted (total planning)
n_towers = 15
max_acquisitions = 40
acquisition_step_size = 10
acquisition_plot_steps = len(range(0,max_acquisitions, acquisition_step_size))
min_towers_acq = 40
towers_per_acq = 10 # need to verify for seq-exp
txs = list(range(0, n_towers*acquisition_plot_steps, n_towers)) # indices into lists 
xs = np.arange(min_towers_acq, \
                min_towers_acq+towers_per_acq*acquisition_step_size*acquisition_plot_steps, \
                towers_per_acq*acquisition_step_size) # number of training towers
##

def plot_tower_stats(logger, mode='nodes expanded'):
    prefix = 'discrete_tower_stats'
    files = [prefix+str(to)+'.pkl' for to in timeouts]

    fig, ax = plt.subplots()
    for file, param in zip(files, timeouts):
        with open(logger.get_figure_path(file), 'rb') as handle:
            tower_stats = pickle.load(handle)
        ts = tower_stats[tower_key]
        median, lower25, upper75 = [], [], []
        for tx in txs:
            # ts_tx should be a vector of n_towers x max_height x timeout/param
            ts_tx = np.array(ts[tx:tx+n_towers]) 
            ts_tx = ts_tx[:,:,-1]  # n_towers x max_height
            nodes_expanded = np.sum(ts_tx, axis=1) # n_towers
            full_towers = ts_tx[:,-1] # n_towers
            if mode=='nodes expanded':
                ts_tx = nodes_expanded
            elif mode=='full towers':
                ts_tx = full_towers
            median.append(np.median(ts_tx))
            lower25.append(np.quantile(ts_tx, 0.25))
            upper75.append(np.quantile(ts_tx, 0.75))
             
        ax.plot(xs, median, label='timeout '+str(param))
        ax.fill_between(xs, lower25, upper75, alpha=0.2)
        
        ax.set_ylabel(mode)
        ax.set_xlabel('Number of training towers')
        ax.legend()
        plt_fname = 'tower_stats_comparison_'+mode+'.png'
        #plt.show()
        plt.savefig(logger.get_figure_path(plt_fname))

    plt.close()
    

def plot_comp(type, logger):
    if type == 'sequential':
        params = timeouts
        label_pre = 'timeout'
        files = ['discrete_sequential_planner_tallest_tower_heights'+str(p)+'.pkl' for p in params]
    elif type == 'total':
        params = n_samples
        label_pre = '# samples'
        files = ['discrete_total_planner_tallest_tower_'+str(p)+'_rewards.pkl' for p in params]    
    
    fig, ax = plt.subplots()
    for file, param in zip(files, params):
        with open(logger.get_figure_path(file), 'rb') as handle:
            heights = pickle.load(handle)

        rs = heights[tower_key]
        median, lower25, upper75 = [], [], []
        for tx in range(len(rs)):
            median.append(np.median(rs[tx]))
            lower25.append(np.quantile(rs[tx], 0.25))
            upper75.append(np.quantile(rs[tx], 0.75))
             
        ax.plot(xs, median, label=label_pre+'='+str(param))
        ax.fill_between(xs, lower25, upper75, alpha=0.2)
         
        ax.set_ylabel('Tower Height')
        ax.set_xlabel('Number of training towers')
        ax.legend()
        plt_fname = type+'_comparison.png'
        plt.savefig(logger.get_figure_path(plt_fname))

    plt.close()
    
def plot_value_stats(logger):
    files = ['discrete_node_values'+str(to)+'.pkl' for to in timeouts]
    
    for file in files:
        with open(logger.get_figure_path(file), 'rb') as handle:
            all_node_values = pickle.load(handle)
            
        for nti, nt in enumerate(xs):
            for tn in range(n_towers):
                if nt == 40 and tn == 0:
                    print(all_node_values[nti][tn])
                plot_values = all_node_values[nti][tn]
                
                fig, ax = plt.subplots()
                for height in plot_values:
                    plot_xs = list(range(1, len(plot_values[height]['median'])+1))
                    ax.plot(plot_xs, plot_values[height]['median'], label=str(height)+' blocks')
                    ax.fill_between(plot_xs, plot_values[height]['25'], plot_values[height]['75'], alpha=0.2)
                ax.set_title('Median Node Values for Different Numbers of Blocks\nin Towers after '+str(nt)+' Training Towers')
                ax.set_ylabel('Median Node Value')
                ax.set_xlabel('MCTS Iteration')
                ax.legend()
                plt_fname = 'node_values_'+str(nt)+'_training_towers_towern'+str(tn)+'.png'
                plt.savefig(logger.get_figure_path(plt_fname))
                plt.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',
                        action='store_true',
                        help='set to run in debug mode')
    parser.add_argument('--exp-path', 
                        type=str, 
                        required=True)
    args = parser.parse_args()
    
    if args.debug:
        import pdb; pdb.set_trace()
    
    logger = ActiveExperimentLogger(args.exp_path)
    
    # plot median height of towers found with sequential method
    plot_comp('sequential', logger)
    # plot median height of towers found with total method
    #plot_comp('total', logger)
    
    # plot number of towers with max_heights blocks found during search
    plot_tower_stats(logger, mode='full towers')    # only works for sequential files and params
    # plot number of nodes in search tree
    plot_tower_stats(logger, mode='nodes expanded') # only works for sequential files and params
    # plot median value of nodes throughout search
    plot_value_stats(logger)                        # only works for sequential files and params
    