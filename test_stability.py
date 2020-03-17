"""
Copyright 2020 Massachusetts Insititute of Technology

Izzy Brand
"""
import sys, os
sys.path.append(os.getcwd() +'/..')

from block_utils import *
from filter_utils import *
from stability import *
import numpy as np
import shutil

def check_stability_with_pybullet(objects, contacts, vis=False, steps=20):
    world = World(objects.values())
    init_positions = get_ps_from_contacts(contacts)
    world.set_poses(init_positions)

    env = Environment([world], vis_sim=vis)
    for t in range(steps):
        env.step(vis_frames=vis)
    env.disconnect()

    final_positions = world.get_positions()
    return unmoved(init_positions, final_positions)

# see if the two dicts of positions are roughly equivalent
def unmoved(init_positions, final_positions, eps=1e-3):
    total_dist = 0
    for obj in init_positions:
        if obj != 'ground':
            total_dist += np.linalg.norm(np.array(init_positions[obj])
                - np.array(final_positions[obj]))
    return total_dist < eps

# compare the results of pybullet with our static analysis
def stable_agrees_with_sim(objects, contacts, vis=False, steps=20):
    simulation_is_stable =\
        check_stability_with_pybullet(objects, contacts, vis=vis, steps=steps)

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

def test_tower_is_stable(vis=False, T=20):
    obj_a = Object('obj_a', Dimensions(1,1,1), 1, Position(0,0,0), Color(0,1,1))
    obj_b = Object('obj_b', Dimensions(3,1,1), 3, Position(0,0,0), Color(1,0,1))
    obj_c = Object('obj_c', Dimensions(1,1,2), 2, Position(0,0,0), Color(1,1,0))

    pos_a_ground = Position(0.,0.,obj_a.dimensions[2]/2)
    con_a_ground = Contact('obj_a', 'ground', pos_a_ground)

    # the single block is stable
    contacts = [con_a_ground]
    objects = {'obj_a': obj_a}
    assert stable_agrees_with_sim(objects, contacts, vis=vis, steps=T)

    # This tower is stable
    stable_pos_b_a = Position(0, 0,
        obj_b.dimensions[2]/2+obj_a.dimensions[2]/2)
    stable_con_b_a = Contact('obj_b', 'obj_a', stable_pos_b_a)
    contacts = [con_a_ground, stable_con_b_a]
    objects = {'obj_a': obj_a, 'obj_b': obj_b}
    assert stable_agrees_with_sim(objects, contacts, vis=vis, steps=T)

    # but if we shift the top block it falls off
    unstable_pos_b_a = Position(1., 0,
        obj_b.dimensions[2]/2+obj_a.dimensions[2]/2)
    unstable_con_b_a = Contact('obj_b', 'obj_a', unstable_pos_b_a)
    contacts = [con_a_ground, unstable_con_b_a]
    objects = {'obj_a': obj_a, 'obj_b': obj_b}
    assert stable_agrees_with_sim(objects, contacts, vis=vis, steps=T)

    # we can stabilize the previously unstable tower by adding a third block
    stable_pos_c_b = Position(-1.4, 0,
        obj_c.dimensions[2]/2+obj_b.dimensions[2]/2)
    stable_con_c_b = Contact('obj_c', 'obj_b', stable_pos_c_b)
    contacts = [con_a_ground, unstable_con_b_a, stable_con_c_b]
    objects = {'obj_a': obj_a, 'obj_b': obj_b, 'obj_c': obj_c}
    assert stable_agrees_with_sim(objects, contacts, vis=vis, steps=T)

    # If we go too far it becomes unstable again
    unstable_pos_c_b = Position(-1.6, 0,
        obj_c.dimensions[2]/2+obj_b.dimensions[2]/2)
    unstable_con_c_b = Contact('obj_c', 'obj_b', unstable_pos_c_b)
    contacts = [con_a_ground, unstable_con_b_a, unstable_con_c_b]
    objects = {'obj_a': obj_a, 'obj_b': obj_b, 'obj_c': obj_c}
    assert stable_agrees_with_sim(objects, contacts, vis=vis, steps=T)

    contacts = [con_a_ground, stable_con_b_a, stable_con_c_b]
    assert stable_agrees_with_sim(objects, contacts, vis=vis, steps=T)

def test_tower_is_constructible():
    obj_a = Object('obj_a', Dimensions(1,1,1), 1, Position(0,0,0), Color(0,1,1))
    obj_b = Object('obj_b', Dimensions(3,1,1), 3, Position(0,0,0), Color(1,0,1))
    obj_c = Object('obj_c', Dimensions(1,1,2), 2, Position(0,0,0), Color(1,1,0))

    pos_a_ground = Position(0.,0.,obj_a.dimensions[2]/2)
    con_a_ground = Contact('obj_a', 'ground', pos_a_ground)

    # the single block is stable
    contacts = [con_a_ground]
    objects = {'obj_a': obj_a}
    assert constructible_agrees_with_stable(objects, contacts)

    # This tower is constructible
    stable_pos_b_a = Position(0.4, 0,
        obj_b.dimensions[2]/2+obj_a.dimensions[2]/2)
    stable_con_b_a = Contact('obj_b', 'obj_a', stable_pos_b_a)
    contacts = [con_a_ground, stable_con_b_a]
    objects = {'obj_a': obj_a, 'obj_b': obj_b}
    assert constructible_agrees_with_stable(objects, contacts)

    # but if we shift the top block it falls off
    unstable_pos_b_a = Position(1., 0,
        obj_b.dimensions[2]/2+obj_a.dimensions[2]/2)
    unstable_con_b_a = Contact('obj_b', 'obj_a', unstable_pos_b_a)
    contacts = [con_a_ground, unstable_con_b_a]
    objects = {'obj_a': obj_a, 'obj_b': obj_b}
    assert constructible_agrees_with_stable(objects, contacts)

    # we can stabilize the previously unstable tower by adding a third block,
    # however this tower cannot be constructed
    stable_pos_c_b = Position(-1.4, 0,
        obj_c.dimensions[2]/2+obj_b.dimensions[2]/2)
    stable_con_c_b = Contact('obj_c', 'obj_b', stable_pos_c_b)
    contacts = [con_a_ground, unstable_con_b_a, stable_con_c_b]
    objects = {'obj_a': obj_a, 'obj_b': obj_b, 'obj_c': obj_c}
    assert constructible_agrees_with_stable(objects, contacts)

    # If we center the middle block, the tower becomes constructible
    contacts = [con_a_ground, stable_con_b_a, stable_con_c_b]
    objects = {'obj_a': obj_a, 'obj_b': obj_b, 'obj_c': obj_c}
    assert constructible_agrees_with_stable(objects, contacts)

def test_calc_expected_height(num_samples=100):
    obj_a = Object('obj_a', Dimensions(1,1,1), 1, Position(0,0,0), Color(0,1,1))
    obj_b = Object('obj_b', Dimensions(3,1,1), 3, Position(0,0,0), Color(1,0,1))
    obj_c = Object('obj_c', Dimensions(0.5,0.5,2), 2, Position(0,0,0), Color(1,1,0))
    objects = {'obj_a': obj_a, 'obj_b': obj_b, 'obj_c': obj_c}

    com_filters = {name:
        create_uniform_particles(num_samples, 3, get_com_ranges(obj))
        for name, obj in objects.items()}

    pos_b_ground = Position(0.,0.,obj_b.dimensions[2]/2)
    con_b_ground = Contact('obj_b', 'ground', pos_b_ground)
    pos_c_ground = Position(0.,0.,obj_c.dimensions[2]/2)
    con_c_ground = Contact('obj_c', 'ground', pos_c_ground)
    pos_a_b = Position(0.,0.,obj_a.dimensions[2]/2+obj_b.dimensions[2]/2)
    con_a_b = Contact('obj_a', 'obj_b', pos_a_b)
    pos_b_a = Position(0.,0.,obj_b.dimensions[2]/2+obj_a.dimensions[2]/2)
    con_b_a = Contact('obj_b', 'obj_a', pos_b_a)
    pos_c_a = Position(0.,0.,obj_c.dimensions[2]/2+obj_a.dimensions[2]/2)
    con_c_a = Contact('obj_c', 'obj_a', pos_c_a)
    pos_a_c = Position(0.,0.,obj_a.dimensions[2]/2+obj_c.dimensions[2]/2)
    con_a_c = Contact('obj_a', 'obj_c', pos_a_c)

    # a tower where each block rests fully on the one below is always stable
    contacts = [con_b_ground, con_a_b, con_c_a]
    total_height  = np.sum([obj.dimensions.z for obj in objects.values()])
    assert calc_expected_height(objects, contacts, com_filters) == total_height

    # a tall tower with a skinny block on the bottom is worse than a short
    # tower that can't fall over
    contacts = [con_c_ground, con_a_c, con_b_a]
    tall_and_wobbly = calc_expected_height(objects, contacts, com_filters)
    contacts = [con_b_ground, con_a_b]
    short_and_stable = calc_expected_height(objects, contacts, com_filters)
    assert tall_and_wobbly < short_and_stable

def test_find_tallest_tower():
    obj_a = Object('obj_a', Dimensions(1,1,1), 1, Position(0,0,0), Color(0,1,1))
    obj_b = Object('obj_b', Dimensions(2,2,2), 3, Position(0,0,0), Color(1,0,1))
    obj_c = Object('obj_c', Dimensions(3,3,3), 2, Position(0,0,0), Color(1,1,0))
    obj_d = Object('obj_d', Dimensions(4,4,4), 2, Position(0,0,0), Color(0,0,1))
    objects = {'obj_a': obj_a, 'obj_b': obj_b, 'obj_c': obj_c, 'obj_d': obj_d}

    com_filters = {name:
        create_uniform_particles(100, 3, get_com_ranges(obj))
        for name, obj in objects.items()}

    oracle_tower = ['obj_' + l for l in 'dcba']
    tallest_tower, tallest_contacts = find_tallest_tower(objects, com_filters)
    assert (np.array(tallest_tower) == np.array(oracle_tower)).all()


if __name__ == '__main__':
    test_tower_is_stable(vis=False, T=20)
    test_tower_is_constructible()
    test_calc_expected_height()
    test_find_tallest_tower()
    print('ALL TESTS PASSED')

    # remove temp urdf files (they will accumulate quickly)
    shutil.rmtree('tmp_urdfs')
