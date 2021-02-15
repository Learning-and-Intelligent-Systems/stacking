from block_utils import Object
from learning.domains.towers.generate_tower_training_data import sample_random_tower
from learning.domains.towers.tower_data import TowerDataset, TowerSampler
from tower_planner import TowerPlanner
import pickle
import copy

if __name__ == '__main__':
    with open('learning/domains/towers/eval_block_set_12.pkl', 'rb') as handle:
        blocks = pickle.load(handle)
    tp = TowerPlanner(stability_mode='contains')
    towers = []

    n_stable = 0
    for _ in range(0, 10000):
        copy_blocks = copy.deepcopy(blocks)
        tower, rotated_tower = sample_random_tower(copy_blocks, num_blocks=5, \
                                    ret_rotated=True, discrete=False)
        
        stable = tp.tower_is_constructable(rotated_tower)
        n_stable += stable

    print(f"n_stable: {n_stable}")