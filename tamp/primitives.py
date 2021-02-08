import numpy as np
import random
import pybullet as p

import pb_robot
from pb_robot.tsrs.panda_box import ComputePrePose

from block_utils import rotation_group, ZERO_POS, all_rotations
from pybullet_utils import transformation

from scipy.spatial.transform import Rotation as R

DEBUG_FAILURE = False

def get_grasp_gen(robot, add_slanted_grasps=True):
    # I opt to use TSR to define grasp sets but you could replace this
    # with your favorite grasp generator
    def gen(body):
        # Note, add_slanted_grasps should be True when we're using the platform.
        grasp_tsr = pb_robot.tsrs.panda_box.grasp(body,
            add_slanted_grasps=add_slanted_grasps, add_orthogonal_grasps=True)
        grasps = []

        for sampled_tsr in grasp_tsr:
            grasp_worldF = sampled_tsr.sample()
            grasp_objF = np.dot(np.linalg.inv(body.get_base_link_transform()), grasp_worldF)
            body_grasp = pb_robot.vobj.BodyGrasp(body, grasp_objF, robot.arm)
            grasps.append((body_grasp,))
            # yield (body_grasp,)
        return grasps

    # def gen(body):
    #     dims = body.get_dimensions()


    return gen


def get_stable_gen_table(fixed=[]):
    def gen(body, surface, surface_pos, protation=None):
        """
        Generate a random pose (possibly rotated) on a surface. Rotation
        can be specified for debugging.
        """
        # These poses are useful for regrasping. Poses are more useful for grasping
        # if they are upright.
        dims = body.get_dimensions()

        rotations = all_rotations()#[8:]
        poses = []
        # These are the pre-chosen regrap locations.
        for x, y in [(-0.4, 0.4), (-0.4, -0.4), (0, 0.4)]:
            for rotation in rotations:
                start_pose = body.get_base_link_pose()

                # Get regrasp pose.
                pose = (ZERO_POS, rotation.as_quat())
                body.set_base_link_pose(pose)
                z = pb_robot.placements.stable_z(body, surface)
                pose = ((x, y, z), rotation.as_quat())

                # Check if regrasp pose is valid.
                body.set_base_link_pose(pose)
                if (pose is None) or any(pb_robot.collisions.pairwise_collision(body, b) for b in fixed):
                    body.set_base_link_pose(start_pose)
                    continue
                body.set_base_link_pose(start_pose)

                body_pose = pb_robot.vobj.BodyPose(body, pose)
                poses.append((body_pose,))
        np.random.shuffle(poses)
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


def get_ik_fn(robot, fixed=[], num_attempts=2, approach_frame='gripper', backoff_frame='global', use_wrist_camera=False):
    def fn(body, pose, grasp):
        obstacles = fixed + [body]
        obj_worldF = pb_robot.geometry.tform_from_pose(pose.pose)
        grasp_worldF = np.dot(obj_worldF, grasp.grasp_objF)
        grasp_worldR = grasp_worldF[:3,:3]

        e_x, e_y, e_z = np.eye(3) # basis vectors

        # The x-axis of the gripper points toward the camera
        # The y-axis of the gripper points along the plane of the hand
        # The z-axis of the gripper points forward

        is_top_grasp = grasp_worldR[:,2].dot(-e_z) > 0.999
        is_upside_down_grasp = grasp_worldR[:,2].dot(e_z) > 0.001
        is_gripper_sideways = np.abs(grasp_worldR[:,1].dot(e_z)) > 0.999
        is_camera_down = np.abs(grasp_worldR[:,0].dot(-e_z)) > 0.999
        is_wrist_too_low = grasp_worldF[2,3] < 0.088/2 + 0.005


        if is_gripper_sideways:
            return None
        if is_upside_down_grasp:
            return None
        if is_camera_down:
            return None

        # the gripper is too close to the ground. the wrist of the arm is 88mm
        # in diameter, and it is the widest part of the hand. Include a 5mm
        # clearance
        if not is_top_grasp and is_wrist_too_low:
            return None


        if approach_frame == 'gripper':
            approach_tform = ComputePrePose(grasp_worldF, [0, 0, -0.125], approach_frame)
        elif approach_frame == 'global':
            approach_tform = ComputePrePose(grasp_worldF, [0, 0, 0.125], approach_frame) # Was -0.125
        else:
            raise NotImplementedError()

        if backoff_frame == 'gripper':
            backoff_tform = ComputePrePose(grasp_worldF, [0, 0, -0.125], backoff_frame)
        elif backoff_frame == 'global':
            backoff_tform = ComputePrePose(grasp_worldF, [0, 0, 0.125], backoff_frame) # Was -0.125
        else:
            raise NotImplementedError()

        # if False:
        #     length, lifeTime = 0.2, 0.0

        #     pos, quat = pb_robot.geometry.pose_from_tform(approach_tform)
        #     new_x = transformation([length, 0.0, 0.0], pos, quat)
        #     new_y = transformation([0.0, length, 0.0], pos, quat)
        #     new_z = transformation([0.0, 0.0, length], pos, quat)

        #     p.addUserDebugLine(pos, new_x, [1,0,0], lifeTime=lifeTime, physicsClientId=1)
        #     p.addUserDebugLine(pos, new_y, [0,1,0], lifeTime=lifeTime, physicsClientId=1)
        #     p.addUserDebugLine(pos, new_z, [0,0,1], lifeTime=lifeTime, physicsClientId=1)


        for ax in range(num_attempts):
            q_grasp = robot.arm.ComputeIK(grasp_worldF)
            if (q_grasp is None): continue
            if not robot.arm.IsCollisionFree(q_grasp, obstacles=obstacles):
                return None

            q_approach = robot.arm.ComputeIK(approach_tform, seed_q=q_grasp)
            if (q_approach is None): continue
            if not robot.arm.IsCollisionFree(q_approach, obstacles=obstacles):
                return None
            conf_approach = pb_robot.vobj.BodyConf(robot, q_approach)


            # Only recompute the backoff if it's different from the approach.
            if approach_frame == backoff_frame:
                q_backoff = q_approach
            else:
                q_backoff = robot.arm.ComputeIK(backoff_tform, seed_q=q_grasp)
                if (q_backoff is None): continue
                if not robot.arm.IsCollisionFree(q_backoff, obstacles=obstacles): return None
            conf_backoff = pb_robot.vobj.BodyConf(robot, q_backoff)

            path_approach = robot.arm.snap.PlanToConfiguration(robot.arm, q_approach, q_grasp, obstacles=obstacles)
            path_backoff = robot.arm.snap.PlanToConfiguration(robot.arm, q_grasp, q_backoff, obstacles=obstacles)
            if path_approach is None or path_backoff is None:
                if DEBUG_FAILURE: input('Approach motion failed')
                continue

            command = [pb_robot.vobj.MoveToTouch(robot.arm, q_approach, q_grasp, grasp, body, use_wrist_camera),
                       grasp,
                       pb_robot.vobj.MoveFromTouch(robot.arm, q_backoff)]
            return (conf_approach, conf_backoff, command)
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
        obstacles = assign_fluent_state(fluents)
        fluent_names = [o.get_name() for o in obstacles]
        for o in fixed:
            if o.get_name() not in fluent_names:
                obstacles.append(o)

        path = robot.arm.birrt.PlanToConfiguration(robot.arm, conf1.configuration, conf2.configuration, obstacles=obstacles)

        if path is None:
            if DEBUG_FAILURE: input('Free motion failed')
            return None
        command = [pb_robot.vobj.JointSpacePath(robot.arm, path)]
        return (command,)
    return fn


def get_holding_motion_gen(robot, fixed=[]):
    def fn(conf1, conf2, body, grasp, fluents=[]):
        obstacles = assign_fluent_state(fluents)
        fluent_names = [o.get_name() for o in obstacles]
        for o in fixed:
            if o.get_name() not in fluent_names:
                obstacles.append(o)

        orig_pose = body.get_base_link_pose()
        robot.arm.SetJointValues(conf1.configuration)
        robot.arm.Grab(body, grasp.grasp_objF)

        path = robot.arm.birrt.PlanToConfiguration(robot.arm, conf1.configuration, conf2.configuration, obstacles=obstacles)

        robot.arm.Release(body)
        body.set_base_link_pose(orig_pose)

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
