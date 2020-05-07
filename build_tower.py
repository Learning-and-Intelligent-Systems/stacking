import os
import numpy as np
import time
import csv
import argparse
import pdb
from collections import OrderedDict

import pybullet as p

import pb_robot
import pb_robot.utils_noBase as utils
from pybullet_utils import PyBulletServer
from pybullet_utils import transformation, quat_math, euler_from_quaternion
from transformations import compose_matrix, concatenate_matrices, inverse_matrix,\
                            translation_from_matrix

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
    
def main(args):
    # start pybullet
    utils.connect(use_gui=True)
    utils.disable_real_time()
    utils.set_default_camera()
    utils.enable_gravity()
    
    # Add floor object 
    floor_file = 'models/short_floor.urdf'
    floor = pb_robot.body.createBody(rel_path=floor_file)
    
    # Create robot object 
    robot = pb_robot.panda.Panda()
    
    # tower center 
    tower_offset = (0.3, 0.0, 0.0)
    
    # load object urdf file names, tower poses, and block dimensions from csv
    obj_goal_poses = {}
    obj_dimensions = {}
    poses_path = os.path.join(args.tower_dir, 'obj_poses.csv')
    with open(poses_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            pos, orn = ([float(v) for v in row[1:4]], [float(v) for v in row[4:8]])
            obj_goal_poses[row[0]] = (pos, orn)
            obj_dimensions[row[0]] = [float(d) for d in row[8:11]]
    
    # initialize blocks high up so don't collide with floor or eachother
    obj_dims = {}
    obj_goals = {}
    p_cog_com = {}
    num_blocks = len(obj_goal_poses)
    block_spread = 0.6
    min_block_y = -block_spread/2
    delta_y = block_spread/(num_blocks-1)
    block_y = min_block_y
    block_x = 0.6
    block_z = 0.1  # based on max object size
    block_pos = [block_x, block_y, block_z]
    block_orn = [0., 0., 0., 1.]
    for urdf_name in obj_goal_poses:
        object_urdf = os.path.join(os.path.join(args.tower_dir, 'urdfs'), urdf_name)
        block = pb_robot.body.createBody(abs_path=object_urdf, pose=(block_pos, block_orn))
        block_pos = np.add(block_pos, [0., delta_y, 0.])
        obj_dims[block] = obj_dimensions[urdf_name]

        # obj_goal_poses is the pos of the com, not the center of geometry
        # need to transform and offset base
        pos, quat = obj_goal_poses[urdf_name]
        p_cog_com[block] = p.getDynamicsInfo(block.id, -1)[3]
        p_goal_block_world = transformation(np.multiply(-1, p_cog_com[block]), pos, quat)
        pos = np.add(p_goal_block_world, tower_offset)
        obj_goals[block] = compose_matrix(translate=pos, angles=euler_from_quaternion(quat))
        
    # sort obj_goals by increasing z height
    sorted_obj_goals_tuple = list((k, v) for k, v in sorted(obj_goals.items(), key=lambda item: item[1][2][3]))
    sorted_obj_goals = OrderedDict()
    for body, pose in sorted_obj_goals_tuple:
        sorted_obj_goals[body] = pose

    # let blocks settle
    for t in range(50):
        p.stepSimulation()

    # get transformation of robot hand in EE frame
    p_hand_world, q_hand_world = p.getLinkState(robot.id, 10)[:2]
    trans_hand_world = compose_matrix(translate=p_hand_world, angles=euler_from_quaternion(q_hand_world))
    p_ee_world, q_ee_world = robot.arm.eeFrame.get_link_pose()
    trans_ee_world = compose_matrix(translate=p_ee_world, angles=euler_from_quaternion(q_ee_world))
    p_hand_ee = transformation(p_hand_world, p_ee_world, q_ee_world, inverse=True)
    q_hand_ee = quat_math(q_ee_world, q_hand_world, True, False)
    trans_hand_ee = compose_matrix(translate=p_hand_ee, 
                                    angles=euler_from_quaternion(q_hand_ee))

    # transformation from robot EE to object center of geometry when grabbed (top down grasp)
    # needed when defining Grab()
    trans_block_ee = {}
    for block, dimensions in obj_dims.items():
        trans_block_hand = compose_matrix(translate=[0., 0., dimensions[2]/2], 
                                                    angles=[0., 0., 0.])
        trans_block_ee[block] = concatenate_matrices(trans_block_hand, trans_hand_ee)
                                                    
    # transformation of EE in world frame when grasping block
    grasp_ee_world = {}
    for block, dimensions in obj_dims.items():
        # get pos of center of geometry
        p_com_world, q_block_world = block.get_pose()
        p_block_world = transformation(np.multiply(-1, p_cog_com[block]), p_com_world, q_block_world)
        trans_block_world = compose_matrix(translate=p_block_world,
                                    angles=euler_from_quaternion(q_block_world))
        trans_ee_world = concatenate_matrices(trans_block_world, 
                                                trans_block_ee[block])
        p_ee_world = translation_from_matrix(trans_ee_world)
        grasp_ee_world[block] = compose_matrix(translate=p_ee_world, 
                                                angles=euler_from_quaternion(q_ee_world))

    # build tower
    for block, obj_goal in sorted_obj_goals.items():
        # grasp object
        grasp_pose = grasp_ee_world[block]
        newq = robot.arm.ComputeIK(grasp_pose)
        robot.arm.hand.Open()
        if newq is not None:
            utils.wait_for_user("Move to desired pose?")
            robot.arm.SetJointValues(newq)
        else:
            print("Cannot find valid config")
        robot.arm.hand.Close()
        robot.arm.Grab(block, inverse_matrix(trans_block_ee[block]))
        
        # move block to tower pose
        goal_ee_world = concatenate_matrices(obj_goal, trans_block_ee[block])
        goal_ee_world = compose_matrix(translate=translation_from_matrix(goal_ee_world),
                                        angles=euler_from_quaternion(q_ee_world))
        newq = robot.arm.ComputeIK(goal_ee_world)
        if newq is not None:
            utils.wait_for_user("Move to desired pose?")
            robot.arm.SetJointValues(newq)
        else:
            print("Cannot find valid config")    
        
        # release and open
        utils.wait_for_user("Release obj?")
        robot.arm.Release(block)
        robot.arm.hand.Open()

    utils.wait_for_user("Run sim?")
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
