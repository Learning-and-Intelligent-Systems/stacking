"""
Utilities for planning with PDDLStream
"""

import pb_robot
from pddlstream.language.generator import from_gen_fn, from_list_fn, from_fn, from_test , BoundedGenerator
from pddlstream.utils import read
import tamp.primitives as primitives

def get_pddlstream_info(robot, fixed, movable, add_slanted_grasps, approach_frame, use_vision, home_poses=None):
    """ Gets information for PDDLStream planning problem """
    domain_pddl = read('tamp/domain_stacking.pddl')
    stream_pddl = read('tamp/stream_stacking.pddl')
    constant_map = {}

    fixed = [f for f in fixed if f is not None]
    stream_map = {
        'sample-pose-table': from_list_fn(primitives.get_stable_gen_table(fixed)),
        'sample-pose-home': from_list_fn(primitives.get_stable_gen_home(home_poses, fixed)),
        'sample-pose-block': from_fn(primitives.get_stable_gen_block(fixed)),
        'sample-grasp': from_list_fn(primitives.get_grasp_gen(robot, add_slanted_grasps=True, add_orthogonal_grasps=False)),
        'pick-inverse-kinematics': from_fn(primitives.get_ik_fn(robot, fixed, approach_frame='gripper', backoff_frame='global', use_wrist_camera=use_vision)),
        'place-inverse-kinematics': from_fn(primitives.get_ik_fn(robot, fixed, approach_frame='global', backoff_frame='gripper', use_wrist_camera=False)),
        'plan-free-motion': from_fn(primitives.get_free_motion_gen(robot, fixed)),
        'plan-holding-motion': from_fn(primitives.get_holding_motion_gen(robot, fixed)),
    }

    return domain_pddl, constant_map, stream_pddl, stream_map


def get_initial_pddl_state(robot, pddl_blocks, table, platform_table, platform_leg, frame):
    """
    Get the PDDL representation of the world between experiments. This
    method assumes that all blocks are on the table. We will always "clean
    up" an experiment by moving blocks away from the platform after an
    experiment.
    """
    fixed = [table, platform_table, platform_leg, frame]
    conf = pb_robot.vobj.BodyConf(robot, robot.arm.GetJointValues())
    print('Initial configuration:', conf.configuration)
    init = [('CanMove',),
            ('Conf', conf),
            ('StartConf', conf),
            ('AtConf', conf),
            ('HandEmpty',)]

    table_pose = pb_robot.vobj.BodyPose(table, table.get_base_link_pose())
    init += [('Pose', table, table_pose), 
             ('AtPose', table, table_pose)]

    for body in pddl_blocks:
        print(type(body), body)
        pose = pb_robot.vobj.BodyPose(body, body.get_base_link_pose())
        init += [('Graspable', body),
                ('Pose', body, pose),
                ('AtPose', body, pose),
                ('Block', body),
                ('On', body, table),
                ('Supported', body, pose, table, table_pose)]

    if not platform_table is None:
        platform_pose = pb_robot.vobj.BodyPose(platform_table, platform_table.get_base_link_pose())
        init += [('Pose', platform_table, platform_pose), 
                 ('AtPose', platform_table, platform_pose)]
        init += [('Block', platform_table)]
    init += [('Table', table)]
    return init
