import pdb
# from filter import filter_world
import argparse
from block_utils import Position, Dimensions, Object, World, Color
from actions import make_world
from stability import find_tallest_tower
import numpy as np

def experiment(blocks):
    pass

def display_tower(tallest_tower):
    pass

def main(args):
    # # make world
    # world = make_world([0., 0., 0.])
    # # run filter
    # filter_world(world, args)
    # # find stable arrangement of blocks
    # get a bunch of random blocks
    blocks = [Object.random('obj_%i' % i) for i in range(args.num_blocks)]

    com_filters = experiment(blocks)

    _, tallest_tower = find_tallest_tower(blocks, com_filters)

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
