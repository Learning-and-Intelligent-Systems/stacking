#!/bin/sh

python -m learning.experiments.active_train_towers --block-set-fname="learning/data/may_blocks/blocks/10_random_block_set_2.pkl" \
                                                   --use-latents \
                                                   --strategy subtower \
                                                   --sampler random \
                                                   --n-epochs 100 \
                                                   --exp-name fit-latents \
                                                   --latent-ensemble-exp-path learning/experiments/logs/pretrained_latents-20210527-165656 \
                                                   --latent-ensemble-tx 0 \
                                                   --fit-latents \
                                                   --n-models 7 \
                                                   --n-acquire 2
                                        
