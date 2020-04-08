"""
Copyright 2020 Massachusetts Insititute of Technology

Izzy Brand
"""
import pytest

from stability import *
from block_utils import *

@pytest.fixture()
def vis(pytestconfig):
    return pytestconfig.getoption("vis")

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

def test_stability_tower_is_stable():
    obj_a = Object('a', Dimensions(0.1,0.1,0.1), 1, Position(0,0,0), Color(0,1,1))
    obj_b = Object('b', Dimensions(0.3,0.1,0.1), 3, Position(0,0,0), Color(1,0,1))
    obj_c = Object('c', Dimensions(0.1,0.1,0.2), 2, Position(0,0,0), Color(1,1,0))

    # the single block is stable
    obj_a.pose = Pose(Position(0, 0, 0.05), ZERO_ROT)
    assert tower_is_stable([obj_a])

    # this is stable
    obj_b.pose = Pose(Position(0, 0, 0.15), ZERO_ROT)
    assert tower_is_stable([obj_a, obj_b])

    # this is unstable
    obj_b.pose = Pose(Position(0.06, 0, 0.15), ZERO_ROT)
    assert not tower_is_stable([obj_a, obj_b])

    # it becomes stable when we add another block
    obj_c.pose = Pose(Position(0.0, 0, 0.3), ZERO_ROT)
    assert tower_is_stable([obj_a, obj_b, obj_c])

def test_stability_tower_is_stable_with_sim(vis):
    for _ in range(10):
        # sample three random blocks
        blocks = [Object.random(str(i)) for i in range(3)]
        # stack all the blocks on top of eachother (center of geometry, not COM)
        prev_z = 0
        for block in blocks:
            pos = Position(0,0,block.dimensions.z/2+prev_z)
            block.pose = Pose(pos, ZERO_ROT)
            prev_z += block.dimensions.z

        assert tower_is_stable(blocks) ==\
            tower_is_stable_in_pybullet(blocks, vis=vis, T=50)