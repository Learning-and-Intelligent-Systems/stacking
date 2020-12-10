from copy import copy
from random import choices
from numpy.random import uniform
import numpy as np

import torch
from torch.utils.data import DataLoader

from learning.domains.towers.tower_data import TowerDataset, TowerSampler
from block_utils import all_rotations, get_rotated_block, Quaternion, Pose, ZERO_POS

def random_placement(block, tower):
    all_quaternions = [Quaternion(*o.as_quat()) for o in list(all_rotations())]
    random_orn = choices(all_quaternions, k=1)[0]
    block.pose = Pose(ZERO_POS, random_orn)
    block = get_rotated_block(block)
    # pick random positions for each block
    # figure out how far the block can be moved w/o losing contact w/ the block below
    if len(tower) > 0:
        max_displacement_xy = np.add(tower[-1].dimensions[:2], block.dimensions[:2])/2.
        # randomly sample a displacement
        rel_xy = uniform(max_displacement_xy, -max_displacement_xy)
        # and get the actual positions of the new block
        pos_xy = np.add(tower[-1].pose.pos[:2], rel_xy)
        # calculate the height of the block
        pos_z = tower[-1].pose.pos[2] + block.dimensions[2]
    else:
        pos_xy = ZERO_POS[:2]
        pos_z = block.dimensions[2]
    block.pose = Pose((pos_xy[0], pos_xy[1], pos_z), (0,0,0,1))
    new_tower = copy(tower)
    new_tower.append(block)
    return new_tower
    
def make_tower_dataset(towers):
    tower_vectors = []
    for tower in towers:
        tower_vectors.append([b.vectorize() for b in tower])
    towers_array = np.array(tower_vectors)
    labels = np.zeros((towers_array.shape[0],))
    tower_dict = {}
    k = len(tower)
    key = str(k)+'block'
    tower_dict[key] = {}
    tower_dict[key]['towers'] = towers_array
    tower_dict[key]['labels'] = labels

    # get predictions for each tower.
    tower_dataset = TowerDataset(tower_dict, augment=False)
    tower_sampler = TowerSampler(dataset=tower_dataset,
                                 batch_size=len(towers),
                                 shuffle=False)
    tower_loader = DataLoader(dataset=tower_dataset,
                              batch_sampler=tower_sampler)

    for tensor, _ in tower_loader:
        if torch.cuda.is_available():
            tensor = tensor.cuda()
    return tower_loader