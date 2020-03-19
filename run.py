import pdb
from filter import filter_world
import argparse
from block_utils import Position, Dimensions, Object, World, Color, \
    simulate_from_contacts
from actions import make_platform_world
from stability import find_tallest_tower
import numpy as np


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
    _, tallest_contacts = find_tallest_tower(blocks, com_filters)
    
    # NOTE(izzy): All my code takes a dict {name: Object}, because I wrote it
    # before blocks had name fields. I plan to go back and fix that so it takes
    # lists of block objects instead

    # and visualize the result
    if args.vis:
        simulate_from_contacts(blocks, tallest_contacts, vis=True, T=60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--num_blocks', type=int, default=5)

    args = parser.parse_args()
    if args.debug: pdb.set_trace()

    main(args)
