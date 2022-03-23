import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import re

# towers_per_block = {
#     10: [25, 50, 75, 100, 125, 150, 250, 500, 750, 1000, 1250, 1500],
#     50: [25, 50, 75, 100, 125, 150, 200, 250, 300],
#     100: [25, 50, 75, 100, 125, 150]
# }
towers_per_block = {
    10: [125, 250, 375, 500],
    50: [25, 50, 75, 100],
    100: [25, 50]
}

EXP_DIR = 'learning/experiments/logs'
mode = 'val'
#mode = 'task'
if __name__ == '__main__':
    ALL_REGRETS = {
        10: [],
        50: [],
        100: []
    }
    
    # First populate all regrets where we have eval data ready.
    for num_blocks in ALL_REGRETS.keys():

        for tpb in towers_per_block[num_blocks]:
            regrets = []
            for exp_name in os.listdir(EXP_DIR):
                
                #exp_template = '%dx%d_base-model-random-train_pf-fit-block-.*' % (num_blocks, tpb)
                exp_template = '%dx%d_norot_3d_base-model-random-train_pf-fit-block-.*' % (num_blocks, tpb)

                if re.match(exp_template, exp_name):
                    print('FOUND:', exp_name)
                    #regrets_path = os.path.join(EXP_DIR, exp_name, 'figures', 'cumulative-overhang_-1.00_19_regrets.pkl')
                    if mode == 'task':
                        regrets_path = os.path.join(EXP_DIR, exp_name, 'figures', 'any-overhang_-1.00_19_regrets.pkl')
                    else:
                        regrets_path = os.path.join(EXP_DIR, exp_name, 'figures', 'val_accuracies_19.pkl')
                    
                    if os.path.exists(regrets_path):
                        with open(regrets_path, 'rb') as handle:
                            rs = pickle.load(handle)
                        if mode == 'task':
                            regrets += rs['2block'][0]
                        else:
                            regrets.append(rs['2block'][0])
                    else:
                        print('Not yet evaluated.')
            
            if len(regrets) == 0:
                regrets.append(-0.1)

            ALL_REGRETS[num_blocks].append(regrets)
    
    fig, axes = plt.subplots(1)
    upper75 = {k: [] for k in ALL_REGRETS.keys()}
    median = {k: [] for k in ALL_REGRETS.keys()}
    lower25 = {k: [] for k in ALL_REGRETS.keys()}
    for k in ALL_REGRETS.keys():
        rs = ALL_REGRETS[k]
        for tx in range(len(rs)):
            median[k].append(np.median(rs[tx]))
            lower25[k].append(np.quantile(rs[tx], 0.25))
            upper75[k].append(np.quantile(rs[tx], 0.75))

    for k in ALL_REGRETS.keys():
        #xs = [k * tpb for tpb in towers_per_block[k]]
        xs = [tpb for tpb in towers_per_block[k]]
        axes.plot(xs, median[k], label=k)
        print('Lower:', k, lower25[k])
        print('Median:', k, median[k])
        print('Upper:', k, upper75[k])
        axes.fill_between(xs, lower25[k], upper75[k], alpha=0.2)
        if mode == 'task':
            axes.set_ylim(-0.25, 1.1)
            axes.set_ylabel('Regret on Eval Block')
        else:
            axes.set_ylim(0.75, 1.1)
            axes.set_ylabel('Val Accuracy')
        
        axes.set_xlabel('Number of Training Towers')
        axes.legend()

    plt.show()


