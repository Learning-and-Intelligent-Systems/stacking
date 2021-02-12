import pdb
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pybullet as p
from agents.panda_agent import PandaAgent
from block_utils import Object, Dimensions, Position, Color, get_adversarial_blocks, all_rotations, Pose, ZERO_POS, get_rotated_block, Quaternion
from learning.domains.towers.generate_tower_training_data import sample_random_tower
import pb_robot
import tamp.primitives as primitives

def count_grasp_solutions(agent, block, pose):
    agent.execute()
    fixed = agent.fixed + [b for b in agent.pddl_blocks if b != block]
    fixed = [f for f in fixed if f is not None]

    grasps = primitives.get_grasp_gen(agent.robot, True)(block)
    pick_ik_fn = primitives.get_ik_fn(agent.robot, fixed, approach_frame='gripper', backoff_frame='global', use_wrist_camera=False)
    place_ik_fn = primitives.get_ik_fn(agent.robot, fixed, approach_frame='global', backoff_frame='gripper', use_wrist_camera=False)
    
    n_picks = 0
    n_places = 0
    print(f'Number of grasps: {len(grasps)}')
    for g in grasps:
        pick_sol = pick_ik_fn(block, pose, g[0])
        if pick_sol is not None:
            n_picks += 1
        # place_sol = place_ik_fn(block, pose, g[0])
        # if place_sol is not None:
        #     n_places += 1
        #input('Next grasp?')
        #p.removeAllUserDebugItems(physicsClientId=1)
    print(f'Number of grasps: {len(grasps)}')
    print('Valid pick grasps:', n_picks)
    print('Valid place grasps:', n_places)



def check_tower_position(agent, blocks, base_xy):
    # Build a random tower of blocks.
    n_blocks = 5
    tower_blocks = np.random.choice(blocks, n_blocks, replace=False)
    tower = sample_random_tower(tower_blocks)

    # Simulate this tower at tower_pose.
    base_block = agent.pddl_block_lookup[tower[0].name]
    base_pos = (base_xy[0], base_xy[1], tower[0].pose.pos.z)
    base_pose = (base_pos, tower[0].rotation)
    base_pose = pb_robot.vobj.BodyPose(base_block, base_pose)
    agent.teleport_block(base_block, base_pose.pose)

    placed_tform = pb_robot.geometry.tform_from_pose(base_pose.pose)
    count_grasp_solutions(agent, base_block, base_pose)
    input('Wait')
    # Now loop through the other tower blocks
    for b_ix in range(1, len(tower)):
        bottom_block = tower[b_ix-1]
        bottom_pose = (bottom_block.pose.pos, bottom_block.rotation)
        bottom_tform = pb_robot.geometry.tform_from_pose(bottom_pose)
        top_block = tower[b_ix]
        top_pose = (top_block.pose.pos, top_block.rotation)
        top_tform = pb_robot.geometry.tform_from_pose(top_pose)

        rel_tform = np.linalg.inv(bottom_tform)@top_tform
        top_pddl = agent.pddl_block_lookup[top_block.name]
        bottom_pddl = agent.pddl_block_lookup[bottom_block.name]

        placed_tform = placed_tform@rel_tform
        placed_pose = pb_robot.geometry.pose_from_tform(placed_tform)
        agent.teleport_block(top_pddl, placed_pose)

        pose = pb_robot.vobj.BodyPose(top_pddl, placed_pose)
        count_grasp_solutions(agent, top_pddl, pose)
        input('Wait')

def check_regrasp_position(agent, blocks, base_xy):
    block = [blocks[5]] #np.random.choice(blocks, 1, replace=False)
    tower = sample_random_tower(block)
    # while tower[0].pose.pos.z < 0.04:
    #     tower = sample_random_tower(block)

    # Simulate this tower at tower_pose.
    base_block = agent.pddl_block_lookup[tower[0].name]
    base_pos = (base_xy[0], base_xy[1], tower[0].pose.pos.z)
    base_pose = (base_pos, tower[0].rotation)
    base_pose = pb_robot.vobj.BodyPose(base_block, base_pose)
    agent.teleport_block(base_block, base_pose.pose)

    placed_tform = pb_robot.geometry.tform_from_pose(base_pose.pose)
    count_grasp_solutions(agent, base_block, base_pose)
    input('Wait')

def validate_regrasps(agent, blocks, base_xy):
    blocks = [blocks[0]]

    # For each potential starting rotation: 
    rotations = all_rotations()

    pddl_block = agent.pddl_block_lookup[blocks[0].name]
    for rx in range(len(rotations)):
        rot = rotations[rx]
        q = Quaternion(*rot.as_quat())
        init_pos, init_orn = pddl_block.get_base_link_pose()
        rot_pose = (init_pos, q)
        pddl_block.set_base_link_pose(rot_pose)
        stable_z = pb_robot.placements.stable_z(pddl_block, agent.table)
        
        agent.execute()
        pddl_block.set_base_link_pose(((init_pos[0], init_pos[1], stable_z), q))
        agent.plan()
        pddl_block.set_base_link_pose(((init_pos[0], init_pos[1], stable_z), q))
    
        for gx, goal_rot in enumerate(rotations):
            goal_q = Quaternion(*goal_rot.as_quat())
            blocks[0].pose = Pose(ZERO_POS, goal_q)
            rb = get_rotated_block(blocks[0])
            blocks[0].pose = Pose(Position(0, 0, rb.dimensions.z/2.), goal_q)
            blocks[0].rotation = goal_q
            success, stable = agent.simulate_tower(blocks,
                                        real=False,
                                        base_xy=(0.5, -0.3),
                                        vis=True,
                                        T=2500,
                                        save_tower=False)

            print('Test:', rx, gx, success)
            if not success:
                input('Continue?')

def check_initial_positions(agent, blocks):
    agent.plan()
    for b in blocks:
        block = agent.pddl_block_lookup[b.name]
        pose =  block.get_base_link_pose()
        pose = pb_robot.vobj.BodyPose(block, pose)
        count_grasp_solutions(agent, block, pose)
        input('Wait')

def main(args):
    NOISE=0.00005

    with open(args.blocks_file, 'rb') as handle:
        blocks = pickle.load(handle)[:10]

    agent = PandaAgent(blocks, NOISE,
                       use_platform=False, 
                       teleport=False,
                       use_action_server=False,
                       use_vision=False)

    # check_tower_position(agent, blocks, (0.5, -0.3))
    #check_regrasp_position(agent, blocks, (0.4, 0.4))
    # validate_regrasps(agent, blocks, (-0.4, -0.4))
    check_initial_positions(agent, blocks)
    
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--blocks-file', type=str, default='learning/domains/towers/final_block_set.pkl')
    args = parser.parse_args()

    main(args)
