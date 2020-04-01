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
 * tower_is_constructible
 * calc_expected_height
 * find_tallest_tower
"""


def tower_is_stable(objects, contacts):
    object_names = object_names_in_order(contacts)
    positions = get_poses_from_contacts(contacts)

    # iterate down the tower, checking stability along the way. Is the entire
    # tower above the current block stable on the current block?
    top_total_com = np.array(objects[object_names[-1]].com)
    top_total_mass = objects[object_names[-1]].mass
    top_total_pos = np.array(positions[object_names[-1]].pos)

    # we don't check the top block because there's nothing on top of it, and
    # we don't check the bottom block because it's the ground!
    for obj_name in reversed(object_names[1:-1]):
        obj = objects[obj_name]
        pos = np.array(positions[obj_name].pos)

        # summarize the mass above the current block with an object and a
        # contact. The dimensions and color are None, because this describes
        # multiple objects
        top_total_obj = Object('top_total', None, top_total_mass, top_total_com, None)
        top_total_rel_pose = Pose(top_total_pos - pos, no_rot)
        top_total_contact = Contact(obj_name, 'top_total', top_total_rel_pose)

        # check stability
        if not pair_is_stable(obj, top_total_obj, top_total_contact):
            return False

        # add the obj to the accumulating COM and mass for the top
        new_top_total_mass = obj.mass + top_total_mass
        new_top_total_global_com = ((top_total_com + top_total_pos)\
            * top_total_mass + (obj.com + pos) * obj.mass) \
            / new_top_total_mass
        top_total_com = new_top_total_global_com - pos
        top_total_mass = new_top_total_mass
        top_total_pos = pos

    return True # we've verified the whole tower is stable

def tower_is_constructible(blocks):
    """ Check that each block can be placed on the tower from bottom to top
    without knocking the tower over

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

def pair_is_stable(top, bottom):
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


def calc_expected_height(objects, contacts, com_filters, num_samples=100):
    """ Finds the expected height of a tower

    If we are uncertain about the center of mass of blocks, then for any
    tower there is some probability that it will collapse. This function
    finds the height of a tower times the probability that it is stable

    Arguments:
        objects {dict {str: Object}} -- the objects in the tower
        contacts {list [Contact]} -- the contacts between the tower
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
        sample_objects = {}
        for name, obj in objects.items():
            com = Position(*com_samples[name][i])
            sample_objects[name] = \
                Object(name, obj.dimensions, obj.mass, com, obj.color)

        stable_count += tower_is_stable(sample_objects, contacts) \
            * tower_is_constructible(sample_objects, contacts)

    height = np.sum([obj.dimensions.z for obj in objects.values()])
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
