#!/bin/sh

python -m learning.experiments.active_fit_towers_pf --block-set-fname="learning/data/may_blocks/blocks/10_random_block_set_2.pkl" \
                                                    --exp-name particle-filter \
                                                    --pretrained-ensemble-exp-path learning/experiments/logs/latents-train-marginal-bugfix-20210608-215642 \
                                                    --ensemble-tx 40 \
                                                    --max-acquisitions 50 \
                                                     