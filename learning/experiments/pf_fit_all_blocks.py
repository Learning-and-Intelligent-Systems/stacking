import argparse
import pickle

from learning.active.utils import ActiveExperimentLogger
from learning.experiments.active_fit_towers_pf import run_particle_filter_fitting


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # These need to be specified.
    parser.add_argument('--pretrained-ensemble-exp-path', type=str, required=True, help='Path to a trained ensemble.')
    parser.add_argument('--ensemble-tx', type=int, required=True, help='Timestep of the trained ensemble to evaluate.')
    parser.add_argument('--block-set-fname', type=str, required=True, help='File containing a list of AT LEAST 5 blocks (block_utils.Object) where the block.name is formatted obj_#')
    # These won't change (have good default values).
    parser.add_argument('--strategy', type=str, choices=['bald', 'random', 'task'], default='bald')
    parser.add_argument('--n-samples', type=int, default=1000)
    parser.add_argument('--task', type=str, choices=['overhang', 'tallest', 'min-contact'], default='overhang')
    parser.add_argument('--max-acquisitions', type=int, default=20, help='Number of iterations to run the main active learning loop for.')
    # Our script will change these.
    parser.add_argument('--exp-name', type=str, default='', help='Where results will be saved. Randon number if not specified.')
    parser.add_argument('--eval-block-ixs', nargs='+', type=int, default=[0], help='Indices of which eval blocks to use.')
    args = parser.parse_args()
    args.use_latents = True
    args.fit_pf = True

    with open(args.block_set_fname, 'rb') as handle:
        blocks = pickle.load(handle)

    # Run the PF code for each block.
    for bx in range(len(blocks)):
        train_logger = ActiveExperimentLogger(args.pretrained_ensemble_exp_path, use_latents=True)
        args.exp_name = train_logger.args.exp_name + '_pf-fit-block-%d' % bx
        args.eval_block_ixs = [bx]
        run_particle_filter_fitting(args)

