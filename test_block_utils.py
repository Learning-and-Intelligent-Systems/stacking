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

def test_block_utils_set_stack_poses():
    """ Verify that set_stack_poses is able to create stable towers given lists
    of blocks with known centers of masses
    """
    for num_blocks in range(1,10):
        for _ in range(2):
            # get a list of random blocks
            blocks = [Object.random(str(i)) for i in range(num_blocks)]
            # arange those blocks
            tower = set_stack_poses(blocks)
            # and check their stability
            assert tower_is_stable(tower)
