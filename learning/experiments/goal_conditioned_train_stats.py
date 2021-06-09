import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

from visualize.domains.abc_blocks.performance import plot_error_stats
from learning.experiments.goal_conditioned_train import run_goal_directed_train

def train_stats(args):
    N = 1
    all_trans_error_rates = []
    for n in range(N):
        n_datapoints, trans_error_rates = run_goal_directed_train(args, plot=False)
        all_trans_error_rates.append(trans_error_rates)
        
    plot_error_stats(n_datapoints, all_trans_error_rates)
    
    plt.ion()
    plt.show()
    input('Enter to close plots.')
    plt.close()
    
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
        
    train_stats(args)
    
    '''
    try:
        train_stats(args)
    except:
        import pdb; pdb.post_mortem()
    '''