import matplotlib.pyplot as plt
import numpy as np

from copy import deepcopy

from agents.teleport_agent import TeleportAgent
from block_utils import Object, get_rotated_block
from learning.active.finite.hypotheses import get_all_hypotheses
from learning.generate_tower_training_data import sample_random_tower
from tower_planner import TowerPlanner

REPEATS = 10
MAX_N = 100
NUM_BLOCKS = 5

def entropy(preds):
    p = np.mean(preds)
    if p == 0. or p == 1.:
        return 0.
    return -p*np.log(p) - (1-p)*np.log(1-p)

def find_entropy_tower(blocks, hypotheses, n_samples=250):
    """
    Given a set of candidate hypotheses, find the tower that 
    has the most disagreement between them.

    Currently implemeted as a rejection sampling method that 
    find the highest entropy predictions amongst the models.
    """
    best_tower = None
    max_entropy = -1.

    for _ in range(n_samples):
        num_blocks = np.random.randint(2, len(blocks)+1)
        tower = sample_random_tower(blocks[:num_blocks])
        tower = [get_rotated_block(b) for b in tower]
        tower = [deepcopy(b) for b in tower]

        preds = [h(tower) for h in hypotheses]
        e = entropy(preds)
        if e > max_entropy:
            best_tower = tower
            max_entropy = e
            best_preds = preds

    return best_tower


def active(strategy, vis=False):
    hypotheses = get_all_hypotheses()
    tp = TowerPlanner(stability_mode='contains')

    for nx in range(1, MAX_N):
        # Generate a random set of 5 blocks.
        blocks = [Object.random(f'obj_{ix}') for ix in range(NUM_BLOCKS)]

        # Choose a tower to build.
        if strategy == 'random':
            num_blocks = np.random.randint(2, NUM_BLOCKS+1)
            tower = sample_random_tower(blocks[:num_blocks])
            tower = [get_rotated_block(b) for b in tower]
            tower = [deepcopy(b) for b in tower]
        elif strategy == 'entropy':
            tower = find_entropy_tower(blocks, hypotheses)
        else:
            raise NotImplementedError()

        # Check for consistent models.
        valid_hypotheses = []
        for h in hypotheses:
            true = tp.tower_is_stable(tower)
            pred = h(tower)
            if true == pred:
                valid_hypotheses.append(h)
        hypotheses = valid_hypotheses

        # Visualize the chosen tower and print the updated hypothesis list.
        if vis:
            TeleportAgent.simulate_tower(tower, vis=True, T=300)
            print(hypotheses)

        # Check if true model found.
        if len(hypotheses) == 1:
            break

    return nx

def get_statistics():
    strategies = ['random', 'entropy']
    n_examples = {}
    for s in strategies:
        n_examples[s] = []

    for rx in range(REPEATS):
        print(rx)
        for s in strategies:
            n = active(s)
            n_examples[s].append(n)
    print(n_examples)
    plt.boxplot([n_examples['random'], n_examples['entropy']], labels=['random', 'entropy'])
    plt.xlabel('Active Learning Strategy')
    plt.ylabel('Number of Labeled Towers')
    ax = plt.gca()
    ax.set_yticks(np.arange(0, np.max(n_examples['random'])+5, 5))
    
    plt.grid()
    plt.show()

def example_entropy():
    active('entropy', vis=True)

if __name__ == '__main__':
    get_statistics()
    #example_entropy()
