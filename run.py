import pdb
from filter import filter_world
import argparse
from block_utils import Position, Dimensions, Object, World, Color, \
    simulate_tower
from actions import make_platform_world
from stability import find_tallest_tower
import numpy as np


def main(args):
    # get a bunch of random blocks
    blocks = [Object.random('obj_%i' % i) for i in range(args.num_blocks)]

    # construct a world containing those blocks
    for block in blocks:
        print('Running filter for', block.name)
        world = make_platform_world(block)
        # run the particle filter
        block.com_filter = filter_world(world, args)
    
    # find the tallest tower
    print('Finding tallest tower.')
    tallest_tower = find_tallest_tower(blocks)

    # and visualize the result
    simulate_tower(tallest_tower, vis=True, T=100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--num_blocks', type=int, default=3)

    args = parser.parse_args()
    if args.debug: pdb.set_trace()

    main(args)
