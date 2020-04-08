"""
Copyright 2020 Massachusetts Insititute of Technology

Izzy Brand
"""
from block_utils import *
from filter_utils import *
import numpy as np
from copy import copy
from itertools import permutations, combinations_with_replacement


# TODO(izzy): right now I'm assuming sequential vertical contacts. This get's a
# lot more tricky if we want to place blocks adjacent to eachother


"""
BRAINSTORM/TODO for rewriting stability with quaterions
 * A tower should be a list of blocks
 * the poses of those blocks should be set in global space
 * we should standardize on scipy rotation version of quaternions
"""

def tower_is_stable(tower):
    """ Check that the tower is stable

    NOTE: This function expects blocks in the tower to have zero rotation

    Arguments:
        blocks {List(Object)} -- the tower from bottom to top

    Returns:
        bool -- Whether or not the tower is stable
    """

    top_group = tower[-1]
    # iterate down the tower, checking stability along the way. Is the group of
    # blocks above the current block stable on the current block?
    # we don't check the top block because there's nothing on top of it
    for block in reversed(tower[:-1]):
        # check stability
        if not pair_is_stable(block, top_group):
            return False

        # add the block to the group
        top_group = group_blocks(block, top_group)

    return True # we've verified the whole tower is stable

def tower_is_constructible(blocks):
    """ Check that each block can be placed on the tower from bottom to top
    without knocking the tower over

    NOTE: This function expects blocks in the tower to have zero rotation

    Arguments:
        blocks {List(Object)} -- the tower from bottom to top

    Returns:
        bool -- Whether or not the tower can be built stably
    """
    for i in range(len(blocks) - 1):
        # check that each pair of blocks is stably individually
        top = blocks[i+1]
        bottom = blocks[i]
        if not pair_is_stable(bottom, top): return False

    return True

def pair_is_stable(bottom, top):
    """ Return True if the top object is stable on the bottom object

    Arguments:
        bottom {Object} -- [description]
        top {Object} -- [description]
    """
    # Check if the COM of the top object is within the dimensions of the bottom
    # object. We assume that the two objects are in static planar contact in z,
    # and that the COM of the top object must lie within the object

    top_rel_pos = np.array(top.pose.pos) - np.array(bottom.pose.pos)
    top_rel_com = top_rel_pos + top.com
    return (np.abs(top_rel_com)*2 - bottom.dimensions <= 0)[:2].all()


def calc_expected_height(tower, num_samples=100):
    """ Finds the expected height of a tower

    If we are uncertain about the center of mass of blocks, then for any
    tower there is some probability that it will collapse. This function
    finds the height of a tower times the probability that it is stable.

    NOTE: This funcion modifies the com field of the blocks in tower.
    NOTE: This function expects blocks in the tower to have zero rotation

    Arguments:
        tower {List(Object)} -- the objects in the tower

    Keyword Arguments:
        num_samples {number} -- how many times to sample COM configurations
            (default: {100})

    Returns:
        [float] -- expected height of the tower
    """
    # sample a bunch of COMs for each block in the tower
    com_samples = {block.name:
        sample_particle_distribution(block.com_filter, num_samples)
        for block in tower}
    stable_count = 0

    # for each possible COM sample, check if such a tower would be stable
    for i in range(num_samples):
        for block in tower:
            block.com = com_samples[block.name][i]

        stable_count += tower_is_stable(tower)\
            * tower_is_constructible(tower)

    height = np.sum([block.dimensions.z for block in tower])
    p_stable = stable_count / float(num_samples)
    print('\t\t\t\t', height, stable_count)
    return height * p_stable

def find_tallest_tower(blocks, num_samples=100):
    """ Finds the tallest tower in expectation given uncertainy over COM

    Arguments:
        objects {dict {str: Object}} -- [description]

    Keyword Arguments:
        num_samples {number} -- how many times to sample COM configurations
            (default: {100})

    Returns:
        List(str), List(Contact) -- The names of the objects in th tallest
            tower (from the bottom up), and the contacts between those objects
    """
    n = len(blocks)
    max_height = 0
    max_tower = []
    # for each ordering of blocks
    for tower in permutations(blocks):
        # for each combination of rotations
        for block_orientations in combinations_with_replacement(rotation_group(), r=n):
            # set the orientation of each block in the tower
            for block, orn in zip(tower, block_orientations):
                block.pose = Pose(ZERO_POS, Quaternion(*orn.as_quat()))
            # unrotate the blocks and calculate their poses in the tower
            stacked_tower = set_stack_poses(tower)
            # simulate_tower(stacked_tower, vis=True, T=25)
            # and check the expected height of this particular tower
            height = calc_expected_height(stacked_tower, num_samples=num_samples)
            # save the tallest tower
            if height > max_height:
                max_tower = stacked_tower
                max_height = height

    return max_tower

def set_stack_poses(blocks):
    """ Find the pose of each block if we were to stack the blocks

    Stacks the blocks in the given order (bottom up). Aligns blocks such that
    the center of mass (or mean of the estimated center of mass given a
    com_filter) are all colinear

    NOTE: if the blocks have a rotation, get_rotated_block will be applied to
    each, so the returned blocks have zero rotation

    Arguments:
        blocks {List(Object)} -- the list of blocks in the tower

    Returns:
        blocks {List(Object)} -- the list of blocks in the tower
    """
    # rotate all the blocks (COM, dimensions) by their defined rotations
    blocks = [get_rotated_block(block) for block in blocks]
    prev_z = 0
    for block in blocks:
        pos = np.zeros(3)
        # set the x,y position of the block
        if block.com_filter is not None:
            pos[:2] = -get_mean(block.com_filter)[:2]
        else:
            pos[:2] = -np.array(block.com)[:2]
        # set the relative z position of the block
        pos[2] = prev_z + block.dimensions.z/2
        # and update the block with the desired pose
        block.pose = Pose(Position(*pos), ZERO_ROT)
        # save the height of the top of the block
        prev_z += block.dimensions.z

    return blocks

if __name__ == '__main__':
    blocks = []
    for s in [0.1, 0.2, 0.3]:
        d = np.array([0.1,0.15,0.3])
        blocks.append(Object(str(s), Dimensions(*(d*(s+1))), s**3, Position(0,0,0), Color(1-s,1-s,1-s)))

    for block in blocks:
        block.com_filter = create_uniform_particles(1000, 3, get_com_ranges(block))

    tower = find_tallest_tower(blocks, num_samples=1000)
    simulate_tower(tower, vis=True, T=60)
