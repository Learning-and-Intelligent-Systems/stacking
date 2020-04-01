"""
Copyright 2020 Massachusetts Insititute of Technology

Izzy Brand
"""
from block_utils import *
from filter_utils import *
import numpy as np
from copy import copy
from itertools import permutations


# TODO(izzy): right now I'm assuming sequential vertical contacts. This get's a
# lot more tricky if we want to place blocks adjacent to eachother


"""
BRAINSTORM/TODO for rewriting stability with quaterions
 * A tower should be a list of blocks
 * the poses of those blocks should be set in global space
 * we should standardize on scipy rotation version of quaternions

 Functions to fix
 * calc_expected_height
 * find_tallest_tower
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
        if not pair_is_stable(top, bottom): return False

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


def calc_expected_height(tower, com_filters, num_samples=100):
    """ Finds the expected height of a tower

    If we are uncertain about the center of mass of blocks, then for any
    tower there is some probability that it will collapse. This function
    finds the height of a tower times the probability that it is stable.

    NOTE: This funcion modifies the com field of the blocks in tower.
    NOTE: This function expects blocks in the tower to have zero rotation

    Arguments:
        tower {List(Object)} -- the objects in the tower
        com_filters {dict {str: ParticleDistibution}} -- COM distributions

    Keyword Arguments:
        num_samples {number} -- how many times to sample COM configurations
            (default: {100})

    Returns:
        [float] -- expected height of the tower
    """
    # sample a bunch of COMs for each object in the tower
    com_samples = {obj:
        sample_particle_distribution(com_filters[obj], num_samples)
        for obj in objects}
    stable_count = 0

    # for each possible COM sample, check if such a tower would be stable
    for i in range(num_samples):
        for block in tower:
            block.com = com_samples[block.name]

        stable_count += tower_is_stable(tower) \
            * tower_is_constructible(tower)

    height = np.sum([block.dimensions.z for block in tower])
    return height * stable_count / num_samples

def find_tallest_tower(objects, com_filters, num_samples=100):
    """ Finds the tallest tower in expectation given uncertainy over COM

    Arguments:
        objects {dict {str: Object}} -- [description]
        com_filters {dict {str: ParticleDistibution}} -- COM distributions

    Keyword Arguments:
        num_samples {number} -- how many times to sample COM configurations
            (default: {100})

    Returns:
        List(str), List(Contact) -- The names of the objects in th tallest
            tower (from the bottom up), and the contacts between those objects
    """
    towers = permutations(objects) # all possible block orderings
    max_height = 0
    max_tower = []
    max_contacts = []
    for tower_idx, tower in enumerate(towers):
        # construct the contacts for this block ordering
        contacts = []
        ground_contact_pose = \
            Pose(Position(0, 0, objects[tower[0]].dimensions.z/2), no_rot)
        contacts.append(Contact(tower[0], 'ground', ground_contact_pose))
        for i in range(len(tower)-1):
            name_a = tower[i+1]
            name_b = tower[i]
            mean_x, mean_y, _ =\
                com_filters[name_a].particles.T @ com_filters[name_a].weights
            x_offset = -mean_x
            y_offset = -mean_y
            z_offset = objects[name_a].dimensions.z/2 \
                     + objects[name_b].dimensions.z/2
            contact_pose = Pose(Position(x_offset, y_offset, z_offset), no_rot)
            contacts.append(Contact(name_a, name_b, contact_pose))

        # and check the expected height of this particular tower
        height = calc_expected_height(objects, contacts, com_filters)
        # save the tallest tower
        if height > max_height:
            max_tower = tower
            max_height = height
            max_contacts = contacts

    return max_tower, max_contacts
