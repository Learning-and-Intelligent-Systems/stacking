import os
import numpy as np
import time
import csv
import argparse
import pdb
from collections import OrderedDict
from copy import copy

import pybullet as p

import pb_robot
import pb_robot.utils_noBase as utils
from pybullet_utils import PyBulletServer
from pybullet_utils import transformation, quat_math, euler_from_quaternion, to_pyquat
from transformations import compose_matrix, concatenate_matrices, inverse_matrix,\
                            translation_from_matrix, quaternion_from_matrix

def vis_frame(pos, quat, length=0.2, lifeTime=0.4):
    """ This function visualizes a coordinate frame for the supplied frame where the
    red,green,blue lines correpsond to the x,y,z axes.
    :param p: a vector of length 3, position of the frame (x,y,z)
    :param q: a vector of length 4, quaternion of the frame (x,y,z,w)
    """
    new_x = transformation([length, 0.0, 0.0], pos, quat)
    new_y = transformation([0.0, length, 0.0], pos, quat)
    new_z = transformation([0.0, 0.0, length], pos, quat)

    p.addUserDebugLine(pos, new_x, [1,0,0], lifeTime=lifeTime)
    p.addUserDebugLine(pos, new_y, [0,1,0], lifeTime=lifeTime)
    p.addUserDebugLine(pos, new_z, [0,0,1], lifeTime=lifeTime)

# for the given orientation, return which axis is aligned with the z world axis
def get_up_axis(orn):
    zero_vec = np.zeros(3)
    z_world = [0., 0., 1.]
    up = None
    for dir in range(3):
        unit_vec = copy(zero_vec)
        unit_vec[dir] = 1.0
        axis_vec_world = transformation(unit_vec, zero_vec, orn)
        if abs(np.dot(axis_vec_world, z_world)) == 1.0:
            up = dir
    return up

def main(args):
    # load object urdf file names, tower poses, and block dimensions from csv
    csv_data = []
    poses_path = os.path.join(args.tower_dir, 'obj_poses.csv')
    with open(poses_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            pos = [float(v) for v in row[1:4]]
            orn = [float(v) for v in row[4:8]]
            dims = [float(d) for d in row[8:11]]
            row_info = (row[0], pos, orn, dims)
            csv_data.append(row_info)

    # start pybullet
    utils.connect(use_gui=True)
    utils.disable_real_time()
    utils.set_default_camera()
    utils.enable_gravity()
    
    # Add floor object 
    plane_id = p.loadURDF("plane_files/plane.urdf", (0.,0.,0.))
                         
    # Create robot object and get transform of robot hand in EE frame 
    robot = pb_robot.panda.Panda()
    hand_id = 10
    p_hand_world, q_hand_world = p.getLinkState(robot.id, 10)[:2]
    trans_hand_world = compose_matrix(translate=p_hand_world, angles=euler_from_quaternion(q_hand_world))
    p_ee_world, q_ee_world = robot.arm.eeFrame.get_link_pose()
    trans_ee_world = compose_matrix(translate=p_ee_world, angles=euler_from_quaternion(q_ee_world))
    p_hand_ee = transformation(p_hand_world, p_ee_world, q_ee_world, inverse=True)
    q_hand_ee = quat_math(q_ee_world, q_hand_world, True, False)
    
    # the hand frame is a little inside the hand, want it to be at the point
    # where the hand and finger meet
    p_offset = [0., 0., 0.01]
    trans_hand_ee = compose_matrix(translate=np.add(p_hand_ee, p_offset), 
                                    angles=euler_from_quaternion(q_hand_ee))
    trans_hand_world = concatenate_matrices(trans_ee_world, trans_hand_ee)
    
    # params
    num_blocks = len(csv_data)
    tower_offset = (0.3, 0.0, 0.0)      # tower center in x-y plane
    block_spread_y = 0.6                # width of block initial placement area in y dimension
    delta_y = block_spread_y/num_blocks # distance between blocks in y dimension
    min_y = -block_spread_y/2           # minimum block y dimension
    block_x = 0.6                       # initial position of blocks in x dimension
    
    for n, row in enumerate(csv_data):
        # unpack csv data
        urdf_filename = row[0]
        p_com_goal = row[1]
        q_com_goal = row[2]
        dims = row[3]
        
        # place block in world (on floor at goal orientation)
        up = get_up_axis(q_com_goal)
        block_y = min_y + delta_y*n
        block_z = dims[up]/2
        block_cog_pos = [block_x, block_y, block_z]
        urdf_path = os.path.join(os.path.join(args.tower_dir, 'urdfs'), urdf_filename)
        block = pb_robot.body.createBody(abs_path=urdf_path, pose=(block_cog_pos, q_com_goal))
        trans_com_cog = compose_matrix(translate=p.getDynamicsInfo(block.id, -1)[3], 
                                                angles=[0., 0., 0., 1.])
                                                
        # transformation from robot EE to object com when grasped (top down grasp)
        # needed when defining Grab()
        q_hand_world = p.getLinkState(robot.id, hand_id)[1]
        q_cog_hand = quat_math(q_com_goal, q_hand_world, True, False)
        trans_cog_hand = compose_matrix(translate=[0., 0., dims[up]/2], 
                                        angles=[0., np.pi, np.pi]) #angles=q_cog_hand)
        trans_cog_ee = concatenate_matrices(trans_hand_ee,
                                            trans_cog_hand)
        trans_com_ee = concatenate_matrices(trans_cog_ee,
                                            trans_com_cog)
                                     
        # transformation of block cog in world frame (at initial block position)
        p_com_world, q_com_world = block.get_pose()
        trans_com_world = compose_matrix(translate=p_com_world, angles=q_com_world)
        
        # transformation of ee in world frame (at initial grasp position)
        trans_ee_world_init = concatenate_matrices(trans_com_world, 
                                                trans_com_ee)
                                                
        # grasp block
        robot.arm.hand.Open()
        grasp_q = robot.arm.ComputeIK(trans_ee_world_init)
        if grasp_q is not None:
            utils.wait_for_user("Move to desired pose?")
            robot.arm.SetJointValues(grasp_q)
        else:
            print("Cannot find valid config")
        robot.arm.hand.Close()
        robot.arm.Grab(block, inverse_matrix(trans_com_ee))

        utils.wait_for_user('Just grasped. Going to move to grasp pose. Ready?')
        robot.arm.SetJointValues(grasp_q)
        
        # transformation of EE in world frame when grasping block (at final block position)
        p_com_world_goal = np.add(p_com_goal, tower_offset)
        trans_com_world_goal = compose_matrix(translate=p_com_world_goal, angles=euler_from_quaternion(q_com_goal))
        trans_ee_world_goal = concatenate_matrices(trans_com_world_goal, 
                                                    trans_com_ee) # should be the other way around
        p_ee_world = translation_from_matrix(trans_ee_world_goal)
        grasp_ee_world_final = compose_matrix(translate=p_ee_world, 
                                                angles=euler_from_quaternion(q_ee_world))

        # move block to goal pose
        goal_q = robot.arm.ComputeIK(grasp_ee_world_final)
        if goal_q is not None:
            utils.wait_for_user("Move to desired pose?")
            robot.arm.SetJointValues(goal_q)
        else:
            print("Cannot find valid config")    

        # release
        utils.wait_for_user('Going to release. Ready?')
        robot.arm.hand.Open()
        robot.arm.Release(block)
        
    for _ in range(1000):
        p.stepSimulation()
    utils.wait_for_user("Done?")
    utils.disconnect()  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tower-dir', type=str, required=True, help='name of directory tower saved to')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    if args.debug: pdb.set_trace()

    main(args)
