import pytest

from stability import *
from block_utils import *

def check_stability_with_pybullet(objects, vis=False, steps=30):
    init_poses = [block.pose for block in tower]
    final_poses = simulate_tower(tower, vis=vis, T=steps)
    return unmoved(init_poses, final_poses)

# see if the positions of two lists of objects are roughly equivalent
def pos_unmoved(init_poses, final_poses, eps=3e-3):
    total_dist = 0
    for init_pose, final_pose in zip(init_pose, final_poses):
        total_dist += np.linalg.norm(np.array(init_pose.pos)
            - np.array(final_pose[obj]))
    return total_dist < eps

def test_stability_pair_is_stable():
    a = Object.random('a')
    b = Object.random('b')

    # center the COM of the top object over the bottom object
    stable_a_pos = Position(-a.com.x,
                            -a.com.y,
                            a.dimensions.z/2 + b.dimensions.z)
    a.pose = Pose(stable_a_pos, ZERO_ROT)
    assert pair_is_stable(a, b)

    # center the COM of the top object on the positive x edge of the bottom
    # object, and then move it a tiny bit farther
    unstable_a_pos = Position(b.dimensions.x/2 - a.com.x + 1e-5,
                              -a.com.y,
                              a.dimensions.z/2 + b.dimensions.z)
    a.pose = Pose(unstable_a_pos, ZERO_ROT)
    assert not pair_is_stable(a, b)

def test_stability_tower_is_constructible():
    pass
