import argparse
import pickle
import numpy as np
import sys

from agents.panda_agent import PandaAgent, PandaClientAgent
from learning.active.utils import ActiveExperimentLogger
from block_utils import Object
from learning.evaluate.active_evaluate_towers import tallest_tower_regret_evaluation, \
        longest_overhang_regret_evaluation, min_contact_regret_evaluation

if __name__ == '__main__':
    import time
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', 
                        choices=['tallest', 'overhang', 'min-contact', 'deconstruct'], 
                        default='tallest',
                        help='planning problem/task to plan for')
    parser.add_argument('--block-set-fname', 
                        type=str, 
                        default='',
                        help='path to the block set file. if not set, args.n_blocks random blocks generated.')
    parser.add_argument('--exp-path', 
                        type=str)
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
    parser.add_argument('--tower-sizes',
                        default=[5],
                        type=int,
                        nargs='+',
                        help='number of blocks in goal tower (can do multiple)')
    parser.add_argument('--exec-xy-noise',
                        type=float,
                        help='noise to add to xy position of blocks if exec-mode==noisy-model')
    parser.add_argument('--plan-xy-noise',
                        type=float,
                        help='noise to add to xy position of blocks if plannnig-model==noisy-model')
    parser.add_argument('--exec-mode',
                        type=str,
                        default='simple-model',
                        choices=['simple-model', 'noisy-model', None],
                        help='this is the method used to execute the found plan. If None will just save planned towers and not calculate regret')
    parser.add_argument('--planning-model',
                        type=str,
                        default='learned',
                        choices=['learned', 'noisy-model', 'simple-model'],
                        help='this is the model used at planning time to determine the probability of a tower being stable')
    
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()
 
    assert ((args.acquisition_step is None) and (args.max_acquisitions is not None)) \
            or ((args.max_acquisitions is None) and (args.acquisition_step is not None)), \
            'must set EITHER --aquisition-step OR --max-acquisitions'
     
    if args.planning_model == 'noisy-model' and not args.plan_xy_noise:
        sys.exit('Error: If planning with noisy model, MUST set args.plan_xy_noise')
    
    if args.exec_mode == 'noisy-model' and not args.exec_xy_noise:
        sys.exit('Error: If executing with noisy model, MUST set args.exec_xy_noise')
        
    if args.planning_model == 'learned' and not args.exp_path:
        sys.exit('Error: If planning with learned model, MUST set args.exp_path')
     
    if args.block_set_fname != '':
        with open(args.block_set_fname, 'rb') as f:
            block_set = pickle.load(f)
    else:
        block_set = None

    logger = ActiveExperimentLogger(args.exp_path)
    pre = 'discrete_' if args.discrete else ''
    
    ts_str = ''.join([str(ts) for ts in args.tower_sizes])
    if args.problem == 'tallest':
        fname = pre+'random_planner_tallest_'+ts_str+'_block_towers'
        tallest_tower_regret_evaluation(logger, block_set, fname, args)
    elif args.problem == 'overhang':
        fname = pre+'random_planner_max_overhang_'+ts_str+'_block_towers'
        longest_overhang_regret_evaluation(logger, block_set, fname, args)
    elif args.problem == 'min-contact':
        fname = pre+'random_planner_min_contact_'+ts_str+'_block_towers'
        min_contact_regret_evaluation(logger, block_set, fname, args)
    end = time.time()
    print('Planner Runtime: %f sec' % (end - start))
    