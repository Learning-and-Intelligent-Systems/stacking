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


def main(args):
    # start pybullet
    utils.connect(use_gui=True)
    utils.disable_real_time()
    utils.set_default_camera()
    utils.enable_gravity()
    
    # Add floor object 
    #floor_file = 'models/short_floor.urdf'
    #floor = pb_robot.body.createBody(rel_path=floor_file)
    plane_id = p.loadURDF("plane_files/plane.urdf", (0.,0.,0.))
                        
    # Create robot object 
    robot = pb_robot.panda.Panda()
    
    # tower center 
    tower_offset = (0.3, 0.0, 0.0)
    
    # load object urdf file names, tower poses, and block dimensions from csv
    obj_goal_poses = OrderedDict()
    obj_dimensions = {}
    poses_path = os.path.join(args.tower_dir, 'obj_poses.csv')
    with open(poses_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            pos, orn = ([float(v) for v in row[1:4]], [float(v) for v in row[4:8]])
            obj_goal_poses[row[0]] = (pos, orn)
            obj_dimensions[row[0]] = [float(d) for d in row[8:11]]
    
    # initialize blocks high up so don't collide with floor or eachother
    obj_dims = OrderedDict()
    obj_goals = {}
    trans_com_cog = {}
    num_blocks = len(obj_goal_poses)
    block_spread = 0.6
    min_block_y = -block_spread/2
    delta_y = block_spread/(num_blocks-1)
    block_y = min_block_y
    block_x = 0.6
    for urdf_name in obj_goal_poses:
        # spawn block at final goal orientation, but on the ground
        block_orn = obj_goal_poses[urdf_name][1]
        # find which axis is facing up
        zero_vec = np.zeros(3)
        z_world = [0., 0., 1.]
        up = None
        for dir in range(3):
            unit_vec = copy(zero_vec)
            unit_vec[dir] = 1.0
            axis_vec_world = transformation(unit_vec, zero_vec, block_orn)
            print(np.dot(axis_vec_world, z_world))
            if abs(np.dot(axis_vec_world, z_world)) == 1.0:
                up = dir
        block_z = obj_dimensions[urdf_name][up]/2
        block_pos = [block_x, block_y, block_z]
        object_urdf = os.path.join(os.path.join(args.tower_dir, 'urdfs'), urdf_name)
        block = pb_robot.body.createBody(abs_path=object_urdf, pose=(block_pos, block_orn))
        block_y += delta_y
        obj_dims[block] = obj_dimensions[urdf_name]

        # obj_goal_poses is the pos of the com
        pos, quat = obj_goal_poses[urdf_name]
        trans_com_cog[block] = compose_matrix(translate=p.getDynamicsInfo(block.id, -1)[3], 
                                                angles=[0., 0., 0., 1.])
        #p_goal_block_world = transformation(p_cog_com[block], pos, quat)
        pos = np.add(pos, tower_offset)
        obj_goals[block] = compose_matrix(translate=pos, angles=euler_from_quaternion(quat))

    blocks = tuple(obj_dims.keys())
    
    # let blocks settle
    utils.wait_for_user()
    for t in range(500):
        p.stepSimulation()
        for block in obj_goals:
            pos, orn = p.getBasePositionAndOrientation(block.id)
            #vis_frame(pos, orn, lifeTime=.5)
            #time.sleep(.1)
            #utils.wait_for_user()
    # get transformation of robot hand in EE frame
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
    #pb_robot.viz.draw_pose(pb_robot.geometry.pose_from_tform(trans_hand_world), length=0.5, width=10)
    #utils.wait_for_user()
    
    # transformation from robot EE to object com when grasped (top down grasp)
    # needed when defining Grab()
    trans_com_ee = {}
    # transformation from robot EE to object cog when grasped (top down grasp)
    # needed when defining goal pose of EE
    trans_cog_ee = {}
    for block in blocks:
        trans_cog_hand = compose_matrix(translate=[0., 0., obj_dims[block][2]/2], 
                                        angles=[0., np.pi, np.pi])
        trans_cog_ee[block] = concatenate_matrices(trans_hand_ee,
                                                    trans_cog_hand)
        trans_com_ee[block] = concatenate_matrices(trans_cog_ee[block],
                                                    trans_com_cog[block])
        trans_com_world = concatenate_matrices(trans_ee_world, trans_com_ee[block])
        pb_robot.viz.draw_pose(pb_robot.geometry.pose_from_tform(trans_com_world), length=0.5, width=10)
    pb_robot.viz.draw_pose(pb_robot.geometry.pose_from_tform(np.eye(4)), length=0.5, width=10)
    utils.wait_for_user()
                                 
    # transformation of block cog in world frame (at initial block position)
    trans_cog_world = {}
    for block in blocks:
        p_com_world, q_block_world = block.get_pose()
        trans_com_world = compose_matrix(translate=p_com_world, angles=q_block_world)
        trans_cog_world[block] = concatenate_matrices(trans_com_world, 
                                                        inverse_matrix(trans_com_cog[block]))
        #pb_robot.viz.draw_pose(pb_robot.geometry.pose_from_tform(trans_cog_world[block]), length=0.5, width=10)
        #utils.wait_for_user()
        
    # transformation of EE in world frame when grasping block (at initial block position)
    grasp_ee_world_init = {}
    for block in blocks:
        trans_ee_world = concatenate_matrices(trans_cog_world[block], 
                                                trans_cog_ee[block])
        p_ee_world = translation_from_matrix(trans_ee_world)
        grasp_ee_world_init[block] = compose_matrix(translate=p_ee_world, 
                                                angles=euler_from_quaternion(q_ee_world))
        #pb_robot.viz.draw_pose(pb_robot.geometry.pose_from_tform(grasp_ee_world_init[block]), length=0.5, width=10)
        #utils.wait_for_user()
        
    # transformation of EE in world frame when grasping block (at final block position)
    grasp_ee_world_final = {}
    for block in blocks:
        trans_ee_world = concatenate_matrices(obj_goals[block], 
                                                trans_com_ee[block]) # should be the other way around
        p_ee_world = translation_from_matrix(trans_ee_world)
        grasp_ee_world_final[block] = compose_matrix(translate=p_ee_world, 
                                                angles=euler_from_quaternion(q_ee_world))
        #pb_robot.viz.draw_pose(pb_robot.geometry.pose_from_tform(grasp_ee_world_final[block]), length=0.5, width=10)
        #utils.wait_for_user()
    
    # build tower
    init_q = robot.arm.GetJointValues()
    for block, obj_goal in obj_goals.items():
        # grasp object
        newq = robot.arm.ComputeIK(grasp_ee_world_init[block])
        robot.arm.hand.Open()
        if newq is not None:
            utils.wait_for_user("Move to desired pose?")
            robot.arm.SetJointValues(newq)
        else:
            print("Cannot find valid config")
        robot.arm.hand.Close()
        robot.arm.Grab(block, inverse_matrix(trans_com_ee[block]))

        #utils.wait_for_user('just grabbed')
        #robot.arm.SetJointValues(newq)
        
        # move block to tower pose
        newq = robot.arm.ComputeIK(grasp_ee_world_final[block])
        if newq is not None:
            utils.wait_for_user("Move to desired pose?")
            robot.arm.SetJointValues(newq)
        else:
            print("Cannot find valid config")    
        
        # release and open
        utils.wait_for_user("Release obj?")
        robot.arm.Release(block)
        robot.arm.hand.Open()
        
    # move back to home configuration
    robot.arm.SetJointValues(init_q)
    #print('goal poses')
    #for block, obj_goal in obj_goals.items():
    #    print(obj_goal)
        #block.set_pose((obj_goals[block][:3,3], [0., 0., 0., 1.]))
    # NOTE: still slightly off in the x direction!
    utils.wait_for_user("Run sim?")
    #print('true poses')
    for block in obj_goals:
        print(np.subtract(block.get_pose()[0], tower_offset), block.get_pose()[1])
    for _ in range(1000):
        p.stepSimulation()
    print('set to goal pose')
    for block, goal_pose in obj_goals.items():
        pos = translation_from_matrix(goal_pose)
        trans_quat = quaternion_from_matrix(goal_pose)
        py_quat = to_pyquat(trans_quat)
        print(np.subtract(pos, tower_offset), py_quat)
        block.set_pose((goal_pose[:3,3], py_quat))
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
