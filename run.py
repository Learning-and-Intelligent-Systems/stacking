import pdb
from filter import filter_world
import argparse
from block_utils import Position, Dimensions, Object, World, Color
from actions import make_world
from stability import find_tallest_tower
import numpy as np

def display_tower(tallest_tower):
    # TODO(izzy)
    pass

def main(args):
    # get a bunch of random blocks
    blocks = [Object.random('obj_%i' % i) for i in range(args.num_blocks)]
    # construct a world containing those blocks
    world = World(blocks)
    # run the particle filter
    com_filters = filter_world(world, args)
    # find the tallest tower
    _, tallest_tower = find_tallest_tower(blocks, com_filters)
    # and visualize the result
    diplay_tower(tallest_tower)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--num_blocks', type=int, default=5)

    args = parser.parse_args()
    if args.debug: pdb.set_trace()

    main(args)
