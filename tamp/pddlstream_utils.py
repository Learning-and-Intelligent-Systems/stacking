"""
Utilities for tower stacking planning with PDDLStream
"""

import time
import pb_robot
from pddlstream.language.generator import from_gen_fn, from_list_fn, from_fn, from_test , BoundedGenerator
from pddlstream.algorithms.constraints import PlanConstraints, WILD
from pddlstream.algorithms.focused import solve_focused
from pddlstream.utils import read, INF
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


def pddlstream_plan(pddl_info, init, goal, search_sample_ratio=1.0, max_time=INF):
    """ Main function to plan using PDDLStream """
    # constraints = PlanConstraints(skeletons=get_regrasp_skeleton(), exact=True)
    pddlstream_problem = tuple([*pddl_info, init, goal])

    start = time.time()
    plan, cost, _ = solve_focused(pddlstream_problem,
                            #constraints=constraints,
                            unit_costs=True,
                            success_cost=INF,
                            max_skeletons=2,
                            search_sample_ratio=search_sample_ratio,
                            max_time=max_time,
                            verbose=False)
    # TODO: Try planner= argument https://github.com/caelan/pddlstream/blob/stable/pddlstream/algorithms/downward.py
    duration = time.time() - start
    print('Planning Complete: Time %f seconds' % duration)
    return plan, cost


def get_regrasp_skeletons():
    """ 
    Returns a list of plan skeletons for plans with and without regrasps
    TODO: This currently errors with multiple skeletons. Try this again.
    """
    no_regrasp = []
    no_regrasp += [('move_free', [WILD, '?q0', WILD])]
    no_regrasp += [('pick', ['?b0', WILD, WILD, '?g0', '?q0', '?q1', WILD])]
    no_regrasp += [('move_holding', ['?q1', '?q2', '?b0', '?g0', WILD])]
    no_regrasp += [('place', ['?b0', WILD, WILD, WILD, '?g0', '?q2', WILD, WILD])]

    regrasp = []
    regrasp += [('move_free', [WILD, '?rq0', WILD])]
    regrasp += [('pick', ['?rb0', WILD, WILD, '?rg0', '?rq0', '?rq1', WILD])]
    regrasp += [('move_holding', ['?rq1', '?rq2', '?rb0', '?rg0', WILD])]
    regrasp += [('place', ['?rb0', WILD, WILD, WILD, '?rg0', '?rq2', '?rq3', WILD])]
    regrasp += [('move_free', ['?rq3', '?rq4', WILD])]
    regrasp += [('pick', ['?rb0', WILD, WILD, '?rg1', '?rq4', '?rq5', WILD])]
    regrasp += [('move_holding', ['?rq5', '?rq6', '?rb0', '?rg1', WILD])]
    regrasp += [('place', ['?rb0', WILD, WILD, WILD, '?rg1', '?rq6', WILD, WILD])]

    return [no_regrasp, regrasp]
