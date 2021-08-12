#!/bin/sh

python -m learning.experiments.active_train_towers --block-set-fname="learning/data/may_blocks/blocks/10_random_block_set_2.pkl" \
                                                   --strategy subtower \
                                                   --sampler random \
                                                   --n-epochs 100 \
                                                   --exp-name train-marg-fit-single \
                                                   --pretrained-ensemble-exp-path learning/experiments/logs/latents-train-marginal-bugfix-20210608-215642 \
                                                   --ensemble-tx 40 \
                                                   --fit \
                                                   --com-repr latent \
                                                   --n-models 10 \
                                                   --n-acquire 2 \
                                                   --max-acquisitions 50 