import argparse
import numpy as np
import os
import pickle

from block_utils import Object
from learning.domains.towers.active_utils import sample_unlabeled_data, get_labels


def get_dataset(args):
    with open(args.block_set_fname, 'rb') as handle:
        block_set = pickle.load(handle)

    towers_dict = sample_unlabeled_data(block_set=block_set, n_samples=args.n_towers)
    towers_dict = get_labels(dataset=towers_dict,
                             exec_mode='noisy-model', 
                             agent=None, 
                             logger=None, 
                             xy_noise=0.003, 
                             label_subtowers=True,
                             save_tower=False)
    return towers_dict


def generate_block_set(n_blocks, block_set_fname):
    blocks = [Object.random(name='obj_%d' % bx) for bx in range(n_blocks)]
    vecs = np.stack([o.vectorize() for o in blocks])
    
    np_fname = block_set_fname.replace('.pkl', '.npy')
    np.save(np_fname, vecs)

    with open(block_set_fname, 'wb') as handle:
        pickle.dump(blocks, handle)
        

"""
This file is used to generate a random sequential dataset. It is meant
to shortcut running the active learning loop to generate the entire dataset.
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-blocks', type=int)
    parser.add_argument('--block-set-fname', type=str, required=True)
    parser.add_argument('--n-towers', type=int, required=True)
    parser.add_argument('--output-fname', type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.block_set_fname):
        generate_block_set(args.n_blocks, args.block_set_fname)

    dataset = get_dataset(args)
    for k in dataset.keys():
        print(k, dataset[k]['towers'].shape)

    with open(args.output_fname, 'wb') as handle:
        pickle.dump(dataset, handle)

    
