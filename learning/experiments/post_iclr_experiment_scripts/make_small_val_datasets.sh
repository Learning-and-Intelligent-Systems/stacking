#!/bin/sh

python -m learning.experiments.create_random_sequential_dataset --n-blocks 10 --block-set-fname learning/data/iclr_data/blocks/10_random_block_set_1.pkl --n-towers 250 --output-fname learning/data/iclr_data/train_towers_small/10x125_norot_random_val.pkl --disable-rotations
python -m learning.experiments.create_random_sequential_dataset --n-blocks 10 --block-set-fname learning/data/iclr_data/blocks/10_random_block_set_1.pkl --n-towers 500 --output-fname learning/data/iclr_data/train_towers_small/10x250_norot_random_val.pkl --disable-rotations
python -m learning.experiments.create_random_sequential_dataset --n-blocks 10 --block-set-fname learning/data/iclr_data/blocks/10_random_block_set_1.pkl --n-towers 750 --output-fname learning/data/iclr_data/train_towers_small/10x375_norot_random_val.pkl --disable-rotations
python -m learning.experiments.create_random_sequential_dataset --n-blocks 10 --block-set-fname learning/data/iclr_data/blocks/10_random_block_set_1.pkl --n-towers 1000 --output-fname learning/data/iclr_data/train_towers_small/10x500_norot_random_val.pkl --disable-rotations

python -m learning.experiments.create_random_sequential_dataset --n-blocks 50 --block-set-fname learning/data/iclr_data/blocks/50_random_block_set_1.pkl --n-towers 250 --output-fname learning/data/iclr_data/train_towers_small/50x25_norot_random_val.pkl --disable-rotations
python -m learning.experiments.create_random_sequential_dataset --n-blocks 50 --block-set-fname learning/data/iclr_data/blocks/50_random_block_set_1.pkl --n-towers 500 --output-fname learning/data/iclr_data/train_towers_small/50x50_norot_random_val.pkl --disable-rotations
python -m learning.experiments.create_random_sequential_dataset --n-blocks 50 --block-set-fname learning/data/iclr_data/blocks/50_random_block_set_1.pkl --n-towers 750 --output-fname learning/data/iclr_data/train_towers_small/50x75_norot_random_val.pkl --disable-rotations
# python -m learning.experiments.create_random_sequential_dataset --n-blocks 50 --block-set-fname learning/data/iclr_data/blocks/50_random_block_set_1.pkl --n-towers 1000 --output-fname learning/data/iclr_data/train_towers_small/50x100_norot_random_val.pkl --disable-rotations

python -m learning.experiments.create_random_sequential_dataset --n-blocks 100 --block-set-fname learning/data/iclr_data/blocks/100_random_block_set_1.pkl --n-towers 500 --output-fname learning/data/iclr_data/train_towers_small/100x25_norot_random_val.pkl --disable-rotations
python -m learning.experiments.create_random_sequential_dataset --n-blocks 100 --block-set-fname learning/data/iclr_data/blocks/100_random_block_set_1.pkl --n-towers 1000 --output-fname learning/data/iclr_data/train_towers_small/100x50_norot_random_val.pkl --disable-rotations