import argparse
import pickle
import numpy as np

from learning.active.utils import ActiveExperimentLogger
from block_utils import Object
from learning.evaluate.active_evaluate_towers import tallest_tower_regret_evaluation, \
        longest_overhang_regret_evaluation, min_contact_regret_evaluation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', 
                        choices=['tallest', 'overhang', 'min-contact', 'deconstruct'], 
                        default='tallest',
                        help='planning problem/task to plan for')
    parser.add_argument('--block-set-fname', 
                        type=str, 
                        default='',
                        required = True,
                        help='path to the block set file. if not set, args.n_blocks random blocks generated.')
    parser.add_argument('--exp-path', 
                        type=str, 
                        required=True)
    parser.add_argument('--max-acquisitions',
                        type=int, 
                        help='evaluate from 0 to this acquisition step (use either this or --acquisition-step)')
    parser.add_argument('--acquisition-step',
                        type=int,
                        help='acquisition step to evaluate (use either this or --max-acquisition)')
    parser.add_argument('--n-towers',
                        default = 50,
                        type=int,
                        help = 'number of tall towers to find for each acquisition step')
    parser.add_argument('--debug',
                        action='store_true',
                        help='set to run in debug mode')
    parser.add_argument('--discrete',
                        action='store_true',
                        help='use if you want to ONLY search the space of block orderings and orientations')
    parser.add_argument('--n-samples',
                        default=5000,
                        type=int,
                        help='number of samples to select from in total planning method')
    parser.add_argument('--tower-sizes',
                        default=[5],
                        nargs='+'
                        help='number of blocks in goal tower (can do multiple)')
    
    args = parser.parse_args()
    
    assert ((args.acquisition_step is None) and (args.max_acquisitions is not None)) \
            or ((args.max_acquisitions is None) and (args.acquisition_step is not None)), \
            'must set EITHER --aquisition-step OR --max-acquisitions'
    
    if args.debug:
        import pdb; pdb.set_trace()
 
    with open(args.block_set_fname, 'rb') as f:
        block_set = pickle.load(f)

    logger = ActiveExperimentLogger(args.exp_path)
    pre = 'discrete_' if args.discrete else ''
    
    if args.problem == 'tallest':
        ts_str = [str(ts) for ts in args.tower_sizes]
        fname = pre+'random_planner_tallest'+ts_str+'_block_towers'
        tallest_tower_regret_evaluation(logger, block_set, fname, args)
    elif args.problem == 'overhang':
        fname = pre+'random_planner_max_overhang_'+ts_str+'_block_towers'
        longest_overhang_regret_evaluation(logger, block_set, fname, args)
    elif args.problem == 'min-contact':
        fname = pre+'random_planner_min_contact_'+ts_str+'_block_towers'
        min_contact_regret_evaluation(logger, block_set, fname, args)
    