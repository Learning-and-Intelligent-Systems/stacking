import pickle
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

from learning.active.utils import ActiveExperimentLogger
from block_utils import Object

def visualize_train_constructability(args):
    logger = ActiveExperimentLogger(args.exp_path)
    
    tx = -1
    constructable_towers = []
    while True:
        if tx == -1:
            tower_dataset = logger.load_dataset(0)
            labels = tower_dataset.tower_labels
            acquired_data = {key: {'labels' : labels[key]}  for key in labels.keys()}
        else:
            acquired_data, _ = logger.load_acquisition_data(tx)
        tx += 1

        if not acquired_data:
            break
        
        n_constructable = 0
        n_total = 0
        for th in acquired_data.keys():
            labels = acquired_data[th]['labels']
            for label in labels:
                n_total += 1
                if label:
                    n_constructable += 1
        if tx == 0:
            print('Initial dataset has %d/%d stable towers' % (n_constructable, n_total))
        else:
            constructable_towers.append(n_constructable)
    
    fig, ax = plt.subplots()
    
    ax.bar(list(range(len(constructable_towers))), constructable_towers)
    ax.set_xlabel('Acquisition Step')
    ax.set_ylabel('Constructable Towers')
    ax.set_title('Constructable Towers per Acquisition Step\n(out of %d Acquired Towers)' % len(labels))
    plt.tight_layout()
    plt.savefig(logger.get_figure_path('training_constructable_towers.png'))
    #plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-path',
                        type=str,
                        required=True)                                        
    parser.add_argument('--debug',
                        action='store_true',
                        help='set to run in debug mode')
    
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    visualize_train_constructability(args)