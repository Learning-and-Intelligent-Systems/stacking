"""
Scan through the logs directories for pf runs. Run the final eval if it hasn't been run yet.
"""
import os
import re
import argparse


EXP_DIR = 'learning/experiments/logs'
#REGRET_PATH = 'cumulative-overhang_-1.00_19_regrets.pkl'
REGRET_PATH = 'any-overhang_-1.00_19_regrets.pkl'
if __name__ == '__main__':
    
    for exp_name in os.listdir(EXP_DIR):
        # Check if this matches the ICLR exp name format.
        # if re.match('[0-9]+x[0-9]+_.*_pf-fit-block-.*', exp_name):
        if re.match('[0-9]+x[0-9]+_norot_3d_.*_pf-fit-block-.*', exp_name):
            figures_path = os.path.join(EXP_DIR, exp_name, 'figures')

            if not REGRET_PATH in os.listdir(figures_path):
                print('Evaluating:', exp_name)
                os.system('python -m learning.evaluate.active_evaluate_towers --eval-type task --problem any-overhang --exec-xy-noise 0.003 --n-towers 50 --acquisition-step 19 --exp-path %s --R-unstable -1' % os.path.join(EXP_DIR, exp_name))
            else:
                print('Already evaluated:', exp_name)
