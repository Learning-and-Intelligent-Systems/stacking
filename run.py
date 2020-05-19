import pdb
from filter import filter_block
import argparse
from block_utils import Position, Dimensions, Object, World, Color, \
    simulate_tower
from actions import make_platform_world
from stability import find_tallest_tower
import numpy as np
import matplotlib.pyplot as plt


def get_adversarial_blocks():
    b1 = Object(name='block1',
                dimensions=Dimensions(0.1, 0.3, 0.1),
                mass=1.,
                com=Position(0.045, 0.145, 0),
                color=Color(0, 0, 1))
    b2 = Object(name='block2',
                dimensions=Dimensions(0.05, 0.3, 0.1),
                mass=1.,
                com=Position(-0.024, -0.145, 0),
                color=Color(1, 0, 1))
    b3 = Object(name='block3',
                dimensions=Dimensions(0.05, 0.05, 0.5),
                mass=1.,
                com=Position(0, 0, 0.2),
                color=Color(0, 1, 1))
    b4 = Object(name='block4',
                dimensions=Dimensions(0.3, 0.05, 0.1),
                mass=1.,
                com=Position(-0.145, 0, -0.03),
                color=Color(0, 1, 0))
    return [b1, b2, b3, b4]


def plot_com_error(errors_random, errors_var):

    for tx in range(0, len(errors_var[0][0])):
        err_rand, err_var = 0, 0
        for bx in range(0, len(errors_var)):
            true = np.array(errors_var[bx][1])
            guess_rand = errors_random[bx][0][tx]
            guess_var = errors_var[bx][0][tx]
            err_var += np.linalg.norm(true-guess_var)
            err_rand += np.linalg.norm(true-guess_rand)
        plt.scatter(tx, err_rand/len(errors_var), c='r')
        plt.scatter(tx, err_var/len(errors_var), c='b')
    plt.show()


def test_exploration(args):
    blocks = [Object.random('obj_%i' % i) for i in range(args.num_blocks)]
    
    errors_random, errors_var = [], []
    for block in blocks:
        print('Running filter for block', block.name)
        _, estimates = filter_block(block, 'random', args)
        errors_random.append((estimates, block.com))
        _, estimates = filter_block(block, 'reduce_var', args)
        errors_var.append((estimates, block.com))
    plot_com_error(errors_random, errors_var)


def main(args):
    # get a bunch of random blocks
    blocks = get_adversarial_blocks()

    # construct a world containing those blocks
    for block in blocks:
        print('Running filter for', block.name)
        # run the particle filter
        
        block.com_filter, _ = filter_block(block, 'reduce_var', args)
        ix = np.argmax(block.com_filter.weights)
        print(block.name, np.array(block.com_filter.particles).T@block.com_filter.weights)
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
    
    test_exploration(args)

    # main(args)
