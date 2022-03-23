"""
Scan through the logs directories for pf runs. Run the final eval if it hasn't been run yet.
"""
import os
import re
import argparse


EXP_DIR = 'learning/experiments/logs'
VAL_PATH = 'val_accuracies_19.pkl'
if __name__ == '__main__':
    
    for exp_name in os.listdir(EXP_DIR):
        # Check if this matches the ICLR exp name format.
        # if re.match('[0-9]+x[0-9]+_.*_pf-fit-block-.*', exp_name):
        for n_train in [10, 50, 100]:
            for eval_ix in range(0, 10):
                if re.match('%dx[0-9]+_norot_3d_.*_pf-fit-block-%d-.*' % (n_train, eval_ix), exp_name):
                    figures_path = os.path.join(EXP_DIR, exp_name, 'figures')

                    if not VAL_PATH in os.listdir(figures_path):
                        print('Evaluating:', exp_name)
                        os.system('python -m learning.evaluate.active_evaluate_towers --eval-type val --acquisition-step 19 --exp-path %s --val-dataset-fname learning/data/iclr_data/eval_towers_small/eval_block_%d_%d_dataset.pkl' % (os.path.join(EXP_DIR, exp_name), n_train, eval_ix))
                    else:
                        print('Already evaluated:', exp_name)
                # else:
                #     print('Could not find model:', '%dx[0-9]+_norot_3d_.*_pf-fit-block-%d-.*' % (n_train, eval_ix))
