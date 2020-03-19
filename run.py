import pdb
from filter import filter_world
import argparse
from block_utils import Position, Dimensions, Object, World, Color
from actions import make_world

def main(args):
    if args.debug:
        import pdb; pdb.set_trace()

    # make world
    world = make_world([0., 0., 0.])

    # run filter
    filter_world(world, args)

    # find stable arrangement of blocks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    main(args)
