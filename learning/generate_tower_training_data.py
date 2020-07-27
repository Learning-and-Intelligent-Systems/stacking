""" Massachusetts Institute of Technology

Izzy Brand, 2020
"""
from agents.teleport_agent import TeleportAgent
from block_utils import Object, Quaternion, Pose, ZERO_POS, rotation_group, get_rotated_block
from tower_planner import TowerPlanner

import argparse
import numpy as np
from random import choices as sample_with_replacement

def build_random_tower(blocks):
    num_blocks = len(blocks)
    # pick random orientations for the blocks
    orns = sample_with_replacement(list(rotation_group()), k=num_blocks)
    orns = [Quaternion(*orn.as_quat()) for orn in orns]
    # apply the rotations to each block
    rotated_blocks = []
    for orn, block in zip(orns, blocks):
        block.pose = Pose(ZERO_POS, orn)
        rotated_blocks.append(get_rotated_block(block))

    # pick random positions for each block
    # get the x and y dimensions of each block (after the rotation)
    dims_xy = np.array([rb.dimensions for rb in rotated_blocks])[:,:2]
    # figure out how far each block can be moved w/ losing contact w/ the block below
    max_displacements_xy = (dims_xy[1:] + dims_xy[:1])/2.
    # sample unscaled noise (clip bceause random normal can exceed -1, 1)
    noise_xy = np.clip(0.5*np.random.randn(num_blocks-1, 2), -0.95, 0.95)
    # and scale the noise by the max allowed displacement
    rel_xy = max_displacements_xy * noise_xy
    # place the first block at the origin
    rel_xy = np.vstack([np.zeros([1,2]), rel_xy])
    # and get the actual positions by cumulative sum of the relative positions
    pos_xy = np.cumsum(rel_xy, axis=0)

    # calculate the height of each block
    heights = np.array([rb.dimensions.z for rb in rotated_blocks])
    cumulative_heights = np.cumsum(heights)
    pos_z = heights/2
    pos_z[1:] += cumulative_heights[:-1]

    # apply the positions to each block
    pos_xyz = np.hstack([pos_xy, pos_z[:,None]])
    for pos, orn, block in zip(pos_xyz, orns, blocks):
        block.pose = Pose(pos, orn)

    return blocks

def main(args):
    # init a tower planner for checking stability
    tp = TowerPlanner()
    for _ in range(args.num_towers):
        # pick a random number of blocks
        # num_blocks = np.random.randint(2, 6)
        num_blocks=2
        # generate random blocks
        blocks = [Object.random(f'Obj_{i}') for i in range(num_blocks)]
        # generate a random tower
        tower = build_random_tower(blocks)
        # if the tower is stable, visualize it for debugging
        if tp.tower_is_stable(tower):
            final_poses = TeleportAgent.simulate_tower(tower, vis=True, T=30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-towers', type=int, default=100)
    # parser.add_argument('--output', type=str, required=True, help='where to save')
    args = parser.parse_args()

    main(args)
