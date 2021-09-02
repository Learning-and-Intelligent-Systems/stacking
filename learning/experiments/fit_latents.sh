#!/bin/sh

python -m learning.experiments.active_train_towers --block-set-fname="learning/data/may_blocks/blocks/10_random_block_set_2.pkl" \
                                                   --strategy subtower \
                                                   --sampler random \
                                                   --n-epochs 100 \
                                                   --exp-name train-random-1000-fit-single-epochs100 \
                                                   --pretrained-ensemble-exp-path learning/experiments/logs/random-latents-train-1000-20210818-125351 \
                                                   --ensemble-tx 0 \
                                                   --fit \
                                                   --com-repr latent \
                                                   --n-models 10 \
                                                   --n-acquire 2 \
                                                   --max-acquisitions 50 