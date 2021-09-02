import argparse
import pickle

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

"""
This file is used to generate a random sequential dataset. It is meant
to shortcut running the active learning loop to generate the entire dataset.
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--block-set-fname', type=str, required=True)
    parser.add_argument('--n-towers', type=int, required=True)
    parser.add_argument('--output-fname', type=str, required=True)
    args = parser.parse_args()

    dataset = get_dataset(args)
    for k in dataset.keys():
        print(k, dataset[k]['towers'].shape)

    with open(args.output_fname, 'wb') as handle:
        pickle.dump(dataset, handle)

    
