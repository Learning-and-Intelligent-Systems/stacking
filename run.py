import pdb
from filter import filter_world
import argparse
from block_utils import Position, Dimensions, Object, World, Color
from actions import make_platform_world
from stability import find_tallest_tower
import numpy as np

def display_tower(tallest_tower):
    # TODO(izzy)
    pass

def main(args):
    # get a bunch of random blocks
    blocks = {}
    for i in range(args.num_blocks):
        blocks['obj_%i' % i] = Object.random('obj_%i' % i) 

    # construct a world containing those blocks
    com_filters = {}
    for block_name in blocks:
        print('Running filter for', block_name)
        world = make_platform_world(blocks[block_name])
        # run the particle filter
        com_filters[block_name] = filter_world(world, args)
    
    # find the tallest tower
    print('Finding tallest tower.')
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
