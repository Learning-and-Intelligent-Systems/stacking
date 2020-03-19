import pdb
from filter import filter_world
import argparse
from block_utils import Position, Dimensions, Object, World, Color,
    simulate_from_contacts
from actions import make_world
from stability import find_tallest_tower
import numpy as np


def main(args):
    # get a bunch of random blocks
    blocks = [Object.random('obj_%i' % i) for i in range(args.num_blocks)]
    # construct a world containing those blocks
    world = World(blocks)
    # run the particle filter
    com_filters = filter_world(world, args)

    # NOTE(izzy): All my code takes a dict {name: Object}, because I wrote it
    # before blocks had name fields. I plan to go back and fix that so it takes
    # lists of block objects instead
    block_dict = {b.name: b for b in blocks}

    # find the tallest tower
    _, tallest_contacts = find_tallest_tower(block_dict, com_filters)
    # and visualize the result
    if args.vis:
        simulate_from_contacts(block_dict, tallest_contacts, vis=True, T=60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--num_blocks', type=int, default=5)

    args = parser.parse_args()
    if args.debug: pdb.set_trace()

    main(args)
