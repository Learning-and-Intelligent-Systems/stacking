import numpy
import random
import pybullet as p

import pb_robot
import tamp.misc as misc

from block_utils import rotation_group, ZERO_POS
from pybullet_utils import transformation

from scipy.spatial.transform import Rotation as R

DEBUG_FAILURE = False 

def get_grasp_gen(robot):
    # I opt to use TSR to define grasp sets but you could replace this
    # with your favorite grasp generator
    def gen(body):
        grasp_tsr = pb_robot.tsrs.panda_box.grasp(body)
        grasps = []
        # Only use a top grasp (2, 4) and side grasps (7).
        for top_grasp_ix in range(len(grasp_tsr)):#[2, 4, 7, 6, 0, 1]: 
            sampled_tsr = grasp_tsr[top_grasp_ix]
            grasp_worldF = sampled_tsr.sample()

            grasp_objF = numpy.dot(numpy.linalg.inv(body.get_base_link_transform()), grasp_worldF)
            body_grasp = pb_robot.vobj.BodyGrasp(body, grasp_objF, robot.arm)
            grasps.append((body_grasp,))
            # yield (body_grasp,)
        return grasps
        
    return gen


def get_stable_gen_table(fixed=[]):
    def gen(body, surface, surface_pos, protation=None):
        """
        Generate a random pose (possibly rotated) on a surface. Rotation
        can be specified for debugging.
        """
        all_rotations = list(rotation_group()) + [R.from_euler('zyx', [0., -numpy.pi/2, 0.])]
        while True:
            for rotation in all_rotations[2:]:
                pose = (ZERO_POS, rotation.as_quat())
                pose = pb_robot.placements.sample_placement(body, surface, top_pose=pose) 
                
                start_pose = body.get_base_link_pose()
                body.set_base_link_pose(pose)
                if (pose is None) or any(pb_robot.collisions.pairwise_collision(body, b) for b in fixed):
                    continue
                body.set_base_link_pose(start_pose)
                body_pose = pb_robot.vobj.BodyPose(body, pose)
                yield (body_pose,)
        return poses

    return gen


def get_stable_gen_block(fixed=[]):
    def fn(body, surface, surface_pose, rel_pose):
        """
        @param rel_pose: A homogeneous transformation matrix.
        """
        surface_tform = pb_robot.geometry.tform_from_pose(surface_pose.pose)
        body_tform = surface_tform@rel_pose
        pose = pb_robot.geometry.pose_from_tform(body_tform)
        body_pose = pb_robot.vobj.BodyPose(body, pose)
        return (body_pose,)
    return fn


def get_ik_fn(robot, fixed=[], num_attempts=2):
    def fn(body, pose, grasp):
        obstacles = fixed + [body]
        obj_worldF = pb_robot.geometry.tform_from_pose(pose.pose)
        grasp_worldF = numpy.dot(obj_worldF, grasp.grasp_objF)
        approach_tform = misc.ComputePrePose(grasp_worldF, [0, 0, -0.05])

        if False:
            length, lifeTime = 0.2, 0.0
            
            pos, quat = pb_robot.geometry.pose_from_tform(approach_tform)
            new_x = transformation([length, 0.0, 0.0], pos, quat)
            new_y = transformation([0.0, length, 0.0], pos, quat)
            new_z = transformation([0.0, 0.0, length], pos, quat)

            p.addUserDebugLine(pos, new_x, [1,0,0], lifeTime=lifeTime)
            p.addUserDebugLine(pos, new_y, [0,1,0], lifeTime=lifeTime)
            p.addUserDebugLine(pos, new_z, [0,0,1], lifeTime=lifeTime)

        for _ in range(num_attempts):
            q_approach = robot.arm.ComputeIK(approach_tform)
            if (q_approach is None) or not robot.arm.IsCollisionFree(q_approach, obstacles=obstacles):
                continue
            conf = pb_robot.vobj.BodyConf(robot, q_approach)
            q_grasp = robot.arm.ComputeIK(grasp_worldF, seed_q=q_approach)
            if (q_grasp is None) or not robot.arm.IsCollisionFree(q_grasp, obstacles=obstacles):
                continue

            path = robot.snap.PlanToConfiguration(robot.arm, q_approach, q_grasp, obstacles=obstacles)
            if path is None:
                if DEBUG_FAILURE: input('Approach motion failed')
                continue
            command = [pb_robot.vobj.JointSpacePath(robot.arm, path), grasp,
                       pb_robot.vobj.JointSpacePath(robot.arm, path[::-1])]
            return (conf, command)
        return None
    return fn


def assign_fluent_state(fluents):
    obstacles = []
    for fluent in fluents:
        name, args = fluent[0], fluent[1:]
        if name == 'atpose':
            o, p = args
            obstacles.append(o)
            o.set_base_link_pose(p.pose)
        else:
            raise ValueError(name)
    return obstacles


def get_free_motion_gen(robot, fixed=[]):
    def fn(conf1, conf2, fluents=[]):
        obstacles = fixed + assign_fluent_state(fluents)
        path = robot.birrt.PlanToConfiguration(robot.arm, conf1.configuration, conf2.configuration, obstacles=obstacles)

        if path is None:
            if DEBUG_FAILURE: input('Free motion failed')
            return None
        command = [pb_robot.vobj.JointSpacePath(robot.arm, path)]
        return (command,)
    return fn


def get_holding_motion_gen(robot, fixed=[]):
    def fn(conf1, conf2, body, grasp, fluents=[]):
        obstacles = fixed + assign_fluent_state(fluents)
        path = robot.birrt.PlanToConfiguration(robot.arm, conf1.configuration, conf2.configuration, obstacles=obstacles)

        if path is None:
            if DEBUG_FAILURE: input('Holding motion failed')
            return None
        command = [pb_robot.vobj.JointSpacePath(robot.arm, path)]
        return (command,)
    return fn


def get_movable_collision_test(robot):
    def test(command, body, pose):
        body.set_base_link_pose(pose.pose)
        obstacles = [body]
        print('Checking collisions!\n\n\n')
        print(command)
        for motion in command:
            if type(motion) != pb_robot.vobj.JointSpacePath: continue
            for q in motion.path:
                if not robot.arm.IsCollisionFree(q, obstacles=obstacles):
                    if DEBUG_FAILURE: input('Movable collision')
                    return False
                print('HERE')
        return True
    return test
