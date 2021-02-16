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
from tamp.primitives import get_free_motion_gen, get_ik_fn, get_grasp_gen
from tamp.misc import load_blocks


def main(args):
    NOISE=0.00005

    # get a bunch of random blocks
    blocks = load_blocks(fname=args.blocks_file,
                         num_blocks=args.num_blocks,
                         remove_ixs=[1])

    agent = PandaAgent(blocks, NOISE,
        use_platform=False, teleport=False,
        use_action_server=False,
        use_vision=False, real=True)
    agent.step_simulation(T=1, vis_frames=True, lifeTime=0.)

    agent.plan()
    fixed = [f for f in agent.fixed if f is not None]
    grasps_fn = get_grasp_gen(agent.robot, add_slanted_grasps=False, add_orthogonal_grasps=True)
    path_planner = get_free_motion_gen(agent.robot, fixed)
    ik_fn = get_ik_fn(agent.robot, fixed, approach_frame='gripper', backoff_frame='global', use_wrist_camera=False)

    from franka_interface import ArmInterface
    arm = ArmInterface()
    arm.move_to_neutral()
    start_q = arm.convertToList(arm.joint_angles())
    start_q = pb_robot.vobj.BodyConf(agent.robot, start_q)

    body = agent.pddl_blocks[args.id]
    pose = pb_robot.vobj.BodyPose(body, body.get_base_link_pose())
    for g in grasps_fn(body):
        grasp = g[0]
        # Check that the grasp points straight down.
        obj_worldF = pb_robot.geometry.tform_from_pose(pose.pose)
        grasp_worldF = np.dot(obj_worldF, grasp.grasp_objF)
        grasp_worldR = grasp_worldF[:3,:3]

        e_x, e_y, e_z = np.eye(3)
        is_top_grasp = grasp_worldR[:,2].dot(-e_z) > 0.999

        if not is_top_grasp: continue

        print('Getting IK...')
        approach_q = ik_fn(body, pose, grasp, return_grasp_q=True)[0]
        print('Planning move to path')
        command1 = path_planner(start_q, approach_q)
        print('Planning return home')
        command2 = path_planner(approach_q, start_q)

        agent.execute()
        input('Ready to execute?')
        command1[0][0].simulate(timestep=0.25)
        input('Move back in sim?')
        command2[0][0].simulate(timestep=0.25)

        input('Move to position on real robot?')
        command1[0][0].execute(realRobot=arm)
        input('Reset position on real robot?')
        command2[0][0].execute(realRobot=arm)

        break



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int)
    parser.add_argument('--blocks-file', type=str, default='learning/domains/towers/final_block_set.pkl')
    parser.add_argument('--show-frames', action='store_true')
    args = parser.parse_args()

    main(args)
