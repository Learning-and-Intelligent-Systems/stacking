#!/bin/sh

python -m learning.experiments.active_train_towers --block-set-fname="learning/data/may_blocks/blocks/10_random_block_set_1.pkl" \
                                                   --exp-name base-model-active-learn \
                                                   --batch-size 16 \
                                                   --exec-mode noisy-model \
                                                   --xy-noise 0.003 \
                                                   --max-acquisitions 40 \
                                                   --n-models 10 \
                                                   --model fcgn \
                                                   --n-hidden 64 \
                                                   --n-samples 100000 \
                                                   --com-repr latent \
                                                   --n-acquire 10 \
                                                   --strategy subtower \
                                                   --sampler sequential  