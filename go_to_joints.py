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
import pb_robot


def main(args):
    NOISE=0.00005

    # get a bunch of random blocks
    if args.use_vision:
        with open(args.blocks_file, 'rb') as handle:
            blocks = pickle.load(handle)[:10]
            blocks = [blocks[1], blocks[2]]
    else:
        blocks = get_adversarial_blocks(num_blocks=args.num_blocks)

    agent = PandaAgent(blocks, NOISE,
        use_platform=False, teleport=False,
        use_action_server=args.use_action_server,
        use_vision=args.use_vision)

    agent.execute()

    # q1 = [0.14654724763473187,  0.21575351569801834,  -0.5849719978198809,  -2.2556906748089887,  -0.6774525616969749, 2.550307508491763,  -0.6516565082767167]
    # q2 = [ -0.14025885624634593, 0.7700725373762062,  -0.8396436988360126,  -1.5055244809443935,  0.8368880648676954,  1.5269749777979318,  -1.9275477043905236]

    # q1 = [-1.9857389225545983, -1.308321625545317,  2.0046107570246647, -2.319278960161609, 0.28307999774306025, 2.756969932860798,  2.387599795313007]
    # q2 = [ 1.14995860276477,-1.0715086746146851, 1.821738397538161,  -2.8114018193240318, 1.9648736125191266,  1.671743112918798,  -0.9022175132022459]

    q1 = [ -1.8355401421019457,  1.0773125105481844, 1.7073429337217094, -2.1553081323948593, -1.5797996042351563,  1.5594502235915926, -0.43448741828070747]
    q2 = [-2.486464799947595,  -1.0559115633903473,  1.8381697061523825, -2.6442660035967593, 2.5008998526608957,  1.5902258537646063,  -1.4465391565816694]


    agent.execution_robot.arm.SetJointValues(q1)
    input()
    agent.execution_robot.arm.SetJointValues(q2)
    input()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--num-blocks', type=int, default=4)
    parser.add_argument('--num-towers', type=int, default=100)
    parser.add_argument('--save-tower', action='store_true')
    parser.add_argument('--use-action-server', action='store_true')
    parser.add_argument('--use-vision', action='store_true', help='get block poses from AR tags')
    parser.add_argument('--blocks-file', type=str, default='learning/domains/towers/final_block_set.pkl')
    parser.add_argument('--real', action='store_true', help='run on real robot')
    parser.add_argument('--show-frames', action='store_true')
    args = parser.parse_args()
    if args.debug: pdb.set_trace()

    # test_exploration(args)

    main(args)
