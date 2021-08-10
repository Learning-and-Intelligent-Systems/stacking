#!/bin/sh

python -m learning.experiments.active_train_towers --block-set-fname="learning/data/may_blocks/blocks/10_random_block_set_2.pkl" \
                                                   --use-latents \
                                                   --strategy subtower \
                                                   --sampler random \
                                                   --n-epochs 100 \
                                                   --exp-name fit-latents-2-acquire-100-sample \
                                                   --pretrained-ensemble-exp-path learning/experiments/logs/latent-blocks-20210528-192208 \
                                                   --ensemble-tx 40 \
                                                   --fit \
                                                   --n-models 10 \
                                                   --n-acquire 2 \
                                                   --max-acquisitions 50
