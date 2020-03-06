"""
Copyright 2020 Massachusetts Insititute of Technology

Izzy Brand
"""
from block_utils import *
from filter_utils import *
import numpy as np
from copy import copy

# TODO(izzy): at some point we will want to be able to be able to place a block
# in any orientation. For now we assume the only choice is (x,y) translation

# TODO(izzy): right now I'm assuming sequential vertical contacts. This get's a
# lot more tricky if we want to place blocks adjacent to eachother


def tower_is_stable(objects, contacts):
    object_names = object_names_in_order(contacts)
    positions = get_ps_from_contacts(contacts)

    # iterate down the tower, checking stability along the way. Is the entire
    # tower above the current block stable on the current block?
    top_total_com = np.array(objects[object_names[-1]].com)
    top_total_mass = objects[object_names[-1]].mass
    top_total_pos = np.array(positions[object_names[-1]])

    # we don't check the top block because there's nothing on top of it, and
    # we don't check the bottom block because it's the ground!
    for obj_name in reversed(object_names[1:-1]):
        obj = objects[obj_name]
        pos = np.array(positions[obj_name])

        # summarize the mass above the current block with an object and a
        # contact. The dimensions and color are None, because this describes
        # multiple objects
        top_total_obj = Object(None, top_total_mass, top_total_com, None)
        top_total_contact = Contact(obj_name, 'top_total', top_total_pos - pos)

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

def tower_is_constructible(objects, contacts):
    contact_dict = get_contact_dict(contacts)

    # iterate up the top to check constructability: Can each block be placed on
    # the last?
    prev_obj_name = contact_dict['ground']

    # start with the second object, which is the first non-ground object
    for _ in range(len(contacts)-1):
        # get the name of the object on top of the previous one
        obj_name = contact_dict[prev_obj_name]
        # and get the relevant contact from the list (there should only be one)
        con = [c for c in contacts if c.objectA_name == obj_name][0]

        if not pair_is_stable(objects[prev_obj_name], objects[obj_name], con):
            return False

    return True

def pair_is_stable(bottom_obj, top_obj, contact):
    """ Return True if the top object is stable on the bottom object

    Arguments:
        bottom_obj {Object} -- [description]
        top_obj {Object} -- [description]
        contact {Contact} -- [description]
    """
    # Check if the COM of the top object is within the dimensions of the bottom
    # object. We assume that the two objects are in static planar contact in z,
    # and that the COM of the top object must lie within the object
    top_rel_com = np.array(top_obj.com) + contact.p_a_b
    return (np.abs(top_rel_com)*2 - bottom_obj.dimensions <= 0)[:2].all()


def calc_expected_height(objects, contacts, com_filters, num_samples=100):
    com_samples = {obj: sample_particle_distribution(com_filters, num_samples) \
        for obj in objects}
    stable_count = 0

    for i in range(num_samples):
        sample_objects = {}
        for name, obj in objects.items():
            com = Position(*com_samples[i])
            sample_objects[name] = Object(obj.dimensions, obj.mass, com, obj.color)

        stable_count += tower_is_stable(sample_objects, contacts) \
            * tower_is_constructible(sample_objects, contacts)

    height = np.sum(obj.dimensions.z for obj in objects.values())
    return height * stable_count / num_samples
