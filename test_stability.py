"""
Copyright 2020 Massachusetts Insititute of Technology

Izzy Brand
"""
import sys, os
sys.path.append(os.getcwd() +'/..')

from block_utils import *
from stability import *
import numpy as np

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
