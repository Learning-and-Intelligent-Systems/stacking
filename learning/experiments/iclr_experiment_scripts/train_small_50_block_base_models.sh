#!/bin/sh

python -m learning.experiments.train_towers_single --train-dataset-fname learning/data/iclr_data/train_towers_small/50x25_norot_random_train.pkl --val-dataset-fname learning/data/iclr_data/train_towers_small/50x25_norot_random_val.pkl --n-blocks 50 --block-set-fname learning/data/iclr_data/blocks/50_random_block_set_1.pkl --exp-name 50x25_norot_3d_base-model-random-train --n-epochs 20 --disable-rotations --com-repr latent
# python -m learning.experiments.train_towers_single --train-dataset-fname learning/data/iclr_data/train_towers_small/50x50_random_norot_train.pkl --val-dataset-fname learning/data/iclr_data/train_towers_small/50x50_norot_random_val.pkl --n-blocks 50 --block-set-fname learning/data/iclr_data/blocks/50_random_block_set_1.pkl --exp-name 50x50_norot_3d_base-model-random-train --n-epochs 20 --disable-rotations --com-repr latent
python -m learning.experiments.train_towers_single --train-dataset-fname learning/data/iclr_data/train_towers_small/50x75_norot_random_train.pkl --val-dataset-fname learning/data/iclr_data/train_towers_small/50x75_norot_random_val.pkl --n-blocks 50 --block-set-fname learning/data/iclr_data/blocks/50_random_block_set_1.pkl --exp-name 50x75_norot_3d_base-model-random-train --n-epochs 20 --disable-rotations --com-repr latent
python -m learning.experiments.train_towers_single --train-dataset-fname learning/data/iclr_data/train_towers_small/50x100_norot_random_train.pkl --val-dataset-fname learning/data/iclr_data/train_towers_small/50x100_norot_random_val.pkl --n-blocks 50 --block-set-fname learning/data/iclr_data/blocks/50_random_block_set_1.pkl --exp-name 50x100_norot_3d_base-model-random-train --n-epochs 20 --disable-rotations --com-repr latent
