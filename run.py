import pdb
from filter import filter_block
import argparse
from block_utils import Position, Dimensions, Object, World, Color, \
    simulate_tower
from actions import make_platform_world
from stability import find_tallest_tower
import numpy as np
def get_adversarial_blocks():
    b1 = Object(name='block1',
                dimensions=Dimensions(0.02, 0.06, 0.02),
                mass=1.,
                com=Position(0.045, 0.145, 0),
                color=Color(0, 0, 1))
    b2 = Object(name='block2',
                dimensions=Dimensions(0.01, 0.06, 0.02),
                mass=1.,
                com=Position(-0.024, -0.145, 0),
                color=Color(1, 0, 1))
    b3 = Object(name='block3',
                dimensions=Dimensions(0.01, 0.01, 0.1),
                mass=1.,
                com=Position(0, 0, 0.2),
                color=Color(0, 1, 1))
    b4 = Object(name='block4',
                dimensions=Dimensions(0.06, 0.01, 0.02),
                mass=1.,
                com=Position(-0.145, 0, -0.03),
                color=Color(0, 1, 0))
    return [b1, b2, b3, b4]


def main(args):
    # get a bunch of random blocks
    blocks = [Object.random('obj_%i' % i) for i in range(args.num_blocks)]
    blocks = get_adversarial_blocks()

    # construct a world containing those blocks
    for block in blocks:
        print('Running filter for', block.name)
        # run the particle filter
        block.com_filter = filter_block(block, args)
        ix = np.argmax(block.com_filter.weights)
        print(block.name, block.com_filter.weights[ix], block.com_filter.particles[ix])
    # find the tallest tower
    print('Finding tallest tower.')
    tallest_tower = find_tallest_tower(blocks)

    # and visualize the result
    simulate_tower(tallest_tower, vis=True, T=100, save_tower=args.save_tower)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--num_blocks', type=int, default=3)
    parser.add_argument('--save-tower', action='store_true')

    args = parser.parse_args()
    if args.debug: pdb.set_trace()

    main(args)
