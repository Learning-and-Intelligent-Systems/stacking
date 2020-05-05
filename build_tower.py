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
    
    # Add floor object 
    floor_file = 'models/short_floor.urdf'
    floor = pb_robot.body.createBody(floor_file)
    
    # Create robot object 
    robot = pb_robot.panda.Panda()
    
    # load object poses from urdf with poses from csv
    poses_path = os.path.join(args.tower_dir, 'obj_poses.csv')
    with open(poses_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            pos, orn = ([float(v) for v in row[1:4]], [float(v) for v in row[4:8]])
            pos = np.add(pos, (0,1,0))
            object_urdf = os.path.join(os.path.join(args.tower_dir, 'urdfs'), row[0])
            block = pb_robot.body.createBody(abs_path=object_urdf, pose=(pos, orn))

    # Example functions over robot arm
    q = robot.arm.GetJointValues()
    pose = robot.arm.ComputeFK(q)
    pose[2, 3] -= 0.1
    pb_robot.viz.draw_pose(pb_robot.geometry.pose_from_tform(pose), length=0.5, width=10)
    newq = robot.arm.ComputeIK(pose)

    # shut down pybullet
    utils.wait_for_user()
    for _ in range(1000):
        p.stepSimulation()
    utils.wait_for_user()
    utils.disconnect()  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tower-dir', type=str, required=True, help='name of directory tower saved to')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    if args.debug: pdb.set_trace()

    main(args)
