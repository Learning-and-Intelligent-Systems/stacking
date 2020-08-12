""" Massachusetts Institute of Technology

Izzy Brand, 2020
"""
from agents.teleport_agent import TeleportAgent
from block_utils import World, Environment, Object, Quaternion, Pose, ZERO_POS, rotation_group, get_rotated_block
from tower_planner import TowerPlanner

import argparse
from copy import deepcopy
import numpy as np
import pickle
from random import choices as sample_with_replacement
import time

def vectorize(tower):
    return [b.vectorize() for b in tower]

def sample_random_tower(blocks):
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

def build_tower(blocks, stable=True, pairwise_stable=True, cog_stable=True, vis=False, max_attempts=100):
    """ Build a tower with the specified stability properties.
    :param stable: Overall tower stability.
    :param pairwise_stable: The stability between two consecutive blocks in the tower.
    :param cog_stable: If the tower is stable just by looking at the CoG.
    """
   
    # init a tower planner for checking stability
    tp = TowerPlanner(stability_mode='angle')

    # since the blocks are sampled with replacement from a finite set, the
    # object instances are sometimes identical. we need to deepcopy the blocks
    # one at a time to make sure that they don't share the same instance
    blocks = [deepcopy(block) for block in blocks]

    for _ in range(max_attempts):
        # generate a random tower
        tower = sample_random_tower(blocks)
        # visualize the tower if desired
        if vis: TeleportAgent.simulate_tower(tower, vis=True, T=20)
        # if the tower is stable, visualize it for debugging
        rotated_tower = [get_rotated_block(b) for b in tower]
        # save the tower if it's stable
        if tp.tower_is_stable(rotated_tower) == stable and \
           tp.tower_is_constructible(rotated_tower) == pairwise_stable and \
           tp.tower_is_cog_stable(rotated_tower) == cog_stable: 
            return rotated_tower

    return None

def get_filename(num_towers, use_block_set, block_set_size, suffix):
    # create a filename for the generated data based on the configuration
    block_set_string = f"{block_set_size}block_set" if use_block_set else "random_blocks"
    return f'learning/data/{block_set_string}_(x{num_towers})_{suffix}.pkl'

def main(args, vis_tower=False):
    # This is a dictionary from stable/unstable label to what subsets of [COG_Stable, PW_Stable] to include.
    difficulty_types = {
        0: [[True, True], [False, True]],#[[True, True], [True, False], [False, True], [False, False]],
        1: [[False, False], [True, False]]#[[True, True], [True, False], [False, True], [False, False]]
    }

    # specify the number of towers to generate
    num_towers_per_cat = 2500
    num_towers = num_towers_per_cat * 2 * 2

    # specify whether to use a finite set of blocks, or to generate new blocks
    # for each tower
    use_block_set = False
    # the number of blocks in the finite set of blocks
    block_set_size = 1000
    # generate the finite set of blocks
    if use_block_set:
        block_set = [Object.random(f'obj_{i}') for i in range(block_set_size)]
    # create a vector of stability labels where half are unstable and half are stable
    stability_labels = np.zeros(num_towers, dtype=int)
    stability_labels[num_towers // 2:] = 1

    dataset = {}
    for num_blocks in range(3, args.max_blocks+1):
        vectorized_towers = []
        block_names = []

        for stable in [0, 1]:
            for cog_stable, pw_stable in difficulty_types[stable]:
                # PW Stability is the same as global stability for two blocks.
                if num_blocks == 2 and pw_stable != stable:
                    continue

                count = 0
                while count < num_towers_per_cat:
                    # print the information about the tower we are about to generate
                    stability_type = "stable" if stable else "unstable"
                    stability_type += "/cog_stable" if cog_stable else "/cog_unstable"
                    stability_type += "/pw_stable" if pw_stable else "/pw_unstable"
                    print(f'{count}/{num_towers_per_cat}\t{stability_type} {num_blocks}-block tower')

                    # generate random blocks. Use the block set if specified. otherwise
                    # generate new blocks from scratch. Save the block names if using blocks
                    # from the block set
                    if use_block_set:
                        blocks = sample_with_replacement(block_set, k=num_blocks)
                    else:
                        blocks = [Object.random(f'obj_{ix}') for ix in range(num_blocks)]

                    # generate a random tower
                    tower = build_tower(blocks, 
                                        stable=stable, 
                                        pairwise_stable=pw_stable, 
                                        cog_stable=cog_stable)
                    
                    if tower is None:
                        continue
                    
                    if vis_tower:
                        w = World(tower)
                        env = Environment([w], vis_sim=True, vis_frames=True)
                        print(stability_type)
                        input()
                        for tx in range(240):
                            env.step(vis_frames=False)
                            time.sleep(1/240.)
                        env.disconnect()
                    count += 1
                    # append the tower to the list
                    vectorized_towers.append(vectorize(tower))
                    block_names.append([b.name for b in blocks])

        data = {
            'towers': np.array(vectorized_towers),
            'labels': stability_labels
        }
        if use_block_set:
            data['block_names'] = block_names

        dataset[f'{num_blocks}block'] = data

    # save the generate data
    filename = get_filename(num_towers, use_block_set, block_set_size, args.suffix)
    print('Saving to', filename)
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--output', type=str, required=True, help='where to save')
    parser.add_argument('--max-blocks', type=int, required=True)
    parser.add_argument('--suffix', type=str, default='')
    args = parser.parse_args()

    main(args)
