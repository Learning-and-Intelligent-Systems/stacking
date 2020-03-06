"""
Copyright 2020 Massachusetts Insititute of Technology

Izzy Brand
"""
from block_utils import *
import numpy as np

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

# see if the two dicts of positions are roughly equivalent
def unmoved(init_positions, final_positions, eps=1e-3):
    total_dist = 0
    for obj in init_positions:
        total_dist += np.linalg.norm(np.array(init_positions[obj])
            - np.array(final_positions[obj]))
    return total_dist < eps

# compare the results of pybullet with our static analysis
def stable_agrees_with_sim(objects, contacts, vis=False):
    positions = get_ps_from_contacts(contacts)
    final_positions = render_objects(objects, positions,
        steps=T, vis=vis, cameraDistance=5)[-1]
    simulation_is_stable = unmoved(positions, final_positions)

    print("STABLE" if simulation_is_stable else "UNSTABLE")
    return simulation_is_stable == tower_is_stable(objects, contacts)

# compare stability with constructibility. a constructible structure should be
# stable with each of the blocks added one by one
def constructible_agrees_with_stable(objects, contacts):
    constructible = tower_is_constructible(objects, contacts)

    piecewise_stable = True
    object_names = object_names_in_order(contacts)

    for i in range(len(contacts)):
        sub_names = object_names[:i+2]
        sub_objects = {o_name: objects[o_name] for o_name in sub_names[1:]}
        sub_contacts = [c for c in contacts if c.objectA_name in sub_names[1:]]
        print(i, sub_names, sub_objects, sub_contacts)
        piecewise_stable &= tower_is_stable(sub_objects, sub_contacts)

    print("CONSTRUCTABLE" if piecewise_stable else "UNCONSTRUCTABLE")
    return constructible == piecewise_stable

if __name__ == '__main__':
    T = 20
    vis = False

    obj_a = Object(Dimensions(1,1,1), 1, Position(0,0,0), Color(0,1,1))
    obj_b = Object(Dimensions(3,1,1), 3, Position(0,0,0), Color(1,0,1))
    obj_c = Object(Dimensions(1,1,2), 2, Position(0,0,0), Color(1,1,0))

    pos_a_ground = Position(0.,0.,obj_a.dimensions[2]/2)
    con_a_ground = Contact('obj_a', 'ground', pos_a_ground)

    # the single block is stable
    contacts = [con_a_ground]
    objects = {'obj_a': obj_a}
    assert stable_agrees_with_sim(objects, contacts, vis=vis)
    assert constructible_agrees_with_stable(objects, contacts)

    # This tower is stable
    stable_pos_b_a = Position(0.4, 0,
        obj_b.dimensions[2]/2+obj_a.dimensions[2]/2)
    stable_con_b_a = Contact('obj_b', 'obj_a', stable_pos_b_a)
    contacts = [con_a_ground, stable_con_b_a]
    objects = {'obj_a': obj_a, 'obj_b': obj_b}
    assert stable_agrees_with_sim(objects, contacts, vis=vis)
    assert constructible_agrees_with_stable(objects, contacts)

    # but if we shift the top block it falls off
    unstable_pos_b_a = Position(1., 0,
        obj_b.dimensions[2]/2+obj_a.dimensions[2]/2)
    unstable_con_b_a = Contact('obj_b', 'obj_a', unstable_pos_b_a)
    contacts = [con_a_ground, unstable_con_b_a]
    objects = {'obj_a': obj_a, 'obj_b': obj_b}
    assert stable_agrees_with_sim(objects, contacts, vis=vis)
    assert constructible_agrees_with_stable(objects, contacts)

    # we can stabilize the previously unstable tower by adding a third block
    stable_pos_c_b = Position(-1.4, 0,
        obj_c.dimensions[2]/2+obj_b.dimensions[2]/2)
    stable_con_c_b = Contact('obj_c', 'obj_b', stable_pos_c_b)
    contacts = [con_a_ground, unstable_con_b_a, stable_con_c_b]
    objects = {'obj_a': obj_a, 'obj_b': obj_b, 'obj_c': obj_c}
    assert stable_agrees_with_sim(objects, contacts, vis=vis)
    assert constructible_agrees_with_stable(objects, contacts)

    # If we go too far it becomes unstable again
    unstable_pos_c_b = Position(-1.6, 0,
        obj_c.dimensions[2]/2+obj_b.dimensions[2]/2)
    unstable_con_c_b = Contact('obj_c', 'obj_b', unstable_pos_c_b)
    contacts = [con_a_ground, unstable_con_b_a, unstable_con_c_b]
    objects = {'obj_a': obj_a, 'obj_b': obj_b, 'obj_c': obj_c}
    assert stable_agrees_with_sim(objects, contacts, vis=vis)
    assert constructible_agrees_with_stable(objects, contacts)

    print('ALL TESTS PASSED')
