import os
import numpy as np
import time
import csv
import argparse
import pdb

import pybullet as p

import pb_robot
import pb_robot.utils_noBase as utils
from pybullet_utils import PyBulletServer

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
    
    # load object urdfs and tower poses from csv
    obj_goal_poses = {}
    poses_path = os.path.join(args.tower_dir, 'obj_poses.csv')
    with open(poses_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            pos, orn = ([float(v) for v in row[1:4]], [float(v) for v in row[4:8]])
            obj_goal_poses[row[0]] = (pos, orn)
    
    # initialize blocks high up so don't collide with floor or eachother
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

    # let blocks settle
    for t in range(50):
        p.stepSimulation()

    # grab object
    #robot.arm.Grab(block, np.eye(4))
    
    # Example functions over robot arm
    '''
    q = robot.arm.GetJointValues()
    pose = robot.arm.ComputeFK(q)
    pose[2, 3] -= 0.1
    pb_robot.viz.draw_pose(pb_robot.geometry.pose_from_tform(pose), length=0.5, width=10)
    newq = robot.arm.ComputeIK(pose)
    if newq is not None:
        raw_input("Move to desired pose?")
        robot.arm.SetJointValues(newq)
    '''
        
    # release obj
    #robot.arm.Release(block)

    # shut down pybullet
    utils.wait_for_user()
    utils.disconnect()  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tower-dir', type=str, required=True, help='name of directory tower saved to')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    if args.debug: pdb.set_trace()

    main(args)
