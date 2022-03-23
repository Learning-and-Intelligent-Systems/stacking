"""
Create datasets if they do not exist yet.
"""
import os
import re
import argparse


DATA_DIR = 'learning/data/post_iclr_data'
TOWERS_PER_BLOCK = {
    10: [25, 50, 75, 100, 125, 250, 375, 500, 625, 750, 875, 1000],
    50: [25, 50, 75, 100, 125, 150, 175, 200],
    100: [25, 50, 75, 100]
}


parser = argparse.ArgumentParser()
parser.add_argument('--n-blocks', type=int, choices=[10, 50, 100], required=True)
args = parser.parse_args()


if __name__ == '__main__':

    tpb_list = TOWERS_PER_BLOCK[args.n_blocks]

    for tpb in tpb_list:

        train_cmd = 'python -m learning.experiments.create_random_sequential_dataset --n-blocks %d --block-set-fname learning/data/post_iclr/blocks/%d_random_block_set_1.pkl --n-towers %d --output-fname learning/data/post_iclr/train_towers_small/%dx%d_norot_random_train.pkl --disable-rotations' % (nblocks, nblocks, nblocks*tpb, nblocks, tpb)
        val_cmd = 'python -m learning.experiments.create_random_sequential_dataset --n-blocks %d --block-set-fname learning/data/post_iclr/blocks/%d_random_block_set_1.pkl --n-towers %d --output-fname learning/data/post_iclr/train_towers_small/%dx%d_norot_random_val.pkl --disable-rotations' % (nblocks, nblocks, int(nblocks*tpb*0.2), nblocks, tpb)
        if not os.path.exists('learning/data/post_iclr/train_towers_small/%dx%d_norot_random_train.pkl' % (nblocks, tpb)):
            os.system(train_cmd)
            os.system(val_cmd)
        else:
            print('Already generated:', nblocks, tpb)
