import pytest

from stability import *
from block_utils import *

@pytest.fixture()
def vis(pytestconfig):
    return pytestconfig.getoption("vis")

def stable_agrees_with_sim(tower, vis=False):
    return tower_is_stable(tower) ==\
        tower_is_stable_in_pybullet(tower, vis=vis)

def tower_is_stable_in_pybullet(tower, vis=False, T=30):
    init_poses = {block.name: block.pose for block in tower}
    final_poses = simulate_tower(tower, vis=vis, T=T)
    return pos_unchanged(init_poses, final_poses)

# see if the two dicts of positions are roughly equivalent
def pos_unchanged(init_poses, final_poses, eps=3e-3):
    total_dist = 0
    for obj in init_poses:
        total_dist += np.linalg.norm(np.array(init_poses[obj].pos)
            - np.array(final_poses[obj]))
    return total_dist < eps

def test_stability_pair_is_stable():
    a = Object.random('a')
    b = Object.random('b')

    # center the COM of the top object over the bottom object
    stable_a_pos = Position(-a.com.x,
                            -a.com.y,
                            a.dimensions.z/2 + b.dimensions.z)
    a.pose = Pose(stable_a_pos, ZERO_ROT)
    assert pair_is_stable(b, a)

    # center the COM of the top object on the positive x edge of the bottom
    # object, and then move it a tiny bit farther
    unstable_a_pos = Position(b.dimensions.x/2 - a.com.x + 1e-5,
                              -a.com.y,
                              a.dimensions.z/2 + b.dimensions.z)
    a.pose = Pose(unstable_a_pos, ZERO_ROT)
    assert not pair_is_stable(b, a)

def test_stability_tower_is_constructible():
    pass

def test_tower_is_stable(vis):
    obj_a = Object('a', Dimensions(0.1,0.1,0.1), 1, Position(0,0,0), Color(0,1,1))
    obj_b = Object('b', Dimensions(0.3,0.1,0.1), 3, Position(0,0,0), Color(1,0,1))
    obj_c = Object('c', Dimensions(0.1,0.1,0.2), 2, Position(0,0,0), Color(1,1,0))

    # the single block is stable
    obj_a.pose = Pose(Position(0, 0, 0.05), ZERO_ROT)
    assert stable_agrees_with_sim([obj_a], vis=vis)

    # this is stable
    obj_b.pose = Pose(Position(0, 0, 0.15), ZERO_ROT)
    assert stable_agrees_with_sim([obj_a, obj_b], vis=vis)

    # this is unstable
    obj_b.pose = Pose(Position(0.06, 0, 0.15), ZERO_ROT)
    assert stable_agrees_with_sim([obj_a, obj_b], vis=vis)

    # it becomes stable when we add another block
    obj_c.pose = Pose(Position(0.0, 0, 0.3), ZERO_ROT)
    assert stable_agrees_with_sim([obj_a, obj_b, obj_c], vis=vis)
