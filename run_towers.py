import pdb
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pybullet as p

from actions import plan_action
from agents.teleport_agent import TeleportAgent
from agents.panda_agent import PandaAgent
from block_utils import Object, Dimensions, Position, Color, get_adversarial_blocks
from learning.domains.towers.generate_tower_training_data import sample_random_tower, build_tower
from particle_belief import ParticleBelief
from tower_planner import TowerPlanner
from tamp.misc import load_blocks
import pb_robot


def main(args):
    NOISE=0.00005

    # get a bunch of random blocks
    # if args.use_vision:
    if True:
        blocks = load_blocks(fname=args.blocks_file,
                             num_blocks=args.num_blocks)
    else:
        blocks = get_adversarial_blocks(num_blocks=args.num_blocks)

    agent = PandaAgent(blocks, NOISE,
        use_platform=False, teleport=False,
        use_action_server=args.use_action_server,
        use_vision=args.use_vision,
        real=args.real)

    if args.show_frames:
        agent.step_simulation(T=1, vis_frames=True, lifeTime=0.)
        input('Start building?')
        p.removeAllUserDebugItems()

    for tx in range(0, args.num_towers):
        # Build a random tower out of blocks.
        n_blocks = np.random.randint(2, args.num_blocks + 1)
        tower_blocks = np.random.choice(blocks, n_blocks, replace=False)
        tower = sample_random_tower(tower_blocks)
        #tower = build_tower(tower_blocks, constructable=True, max_attempts=50000)

        # and execute the resulting plan.
        print(f"Starting tower {tx}")
        if args.use_action_server:
            agent.simulate_tower_parallel(tower,
                                          real=args.real,
                                          base_xy=(0.5, -0.3),
                                          vis=True,
                                          T=2500)
        else:
            success, stable = agent.simulate_tower(tower,
                                real=args.real,
                                base_xy=(0.5, -0.3),
                                vis=True,
                                T=2500,
                                save_tower=args.save_tower)
            if not success:
                print('Planner failed.')
        print(f"Finished tower {tx}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--num-blocks', type=int, default=4)
    parser.add_argument('--num-towers', type=int, default=100)
    parser.add_argument('--save-tower', action='store_true')
    parser.add_argument('--use-action-server', action='store_true')
    parser.add_argument('--use-vision', action='store_true', help='get block poses from AR tags')
    parser.add_argument('--blocks-file', type=str, default='learning/domains/towers/final_block_set_10.pkl')
    parser.add_argument('--real', action='store_true', help='run on real robot')
    parser.add_argument('--show-frames', action='store_true')
    args = parser.parse_args()
    if args.debug: pdb.set_trace()

    # test_exploration(args)

    main(args)
