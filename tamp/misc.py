import numpy
import pb_robot
import os
import shutil
import time

import tamp.primitives as primitives

from block_utils import object_to_urdf, Object, Pose, Position, all_rotations, Quaternion, get_rotated_block
from pddlstream.language.generator import from_gen_fn, from_list_fn, from_fn, from_test , BoundedGenerator
from pddlstream.utils import read


class ExecutionFailure(Exception):
    """
    Defines a task execution failure for the robot.
    If fatal is True, manual intervention is needed. Otherwise, the error
    can be recovered by replanning from current state.
    Optionally, a BodyGrasp can be specified to indicate that the
    robot was holding a particular object to replan with that knowledge.
    """
    def __init__(self, reason="", fatal=False, obj_held=None):
        self.reason = reason
        self.fatal = fatal
        self.obj_held = obj_held

    def __str__(self):
        if self.fatal:
            descriptor = "Fatal"
        else:
            descriptor = "Recoverable"
        print_str = descriptor + " execution failure: " + self.reason
        if self.obj_held is not None:
            print_str += f" while holding {self.obj_held.body.readableName}"
        return print_str


def getDirectory():
    '''Get the file path for the location of kinbody
    @return object_path (string) Path to objects folder'''
    package_name = 'tampExample'
    directory = 'models'
    objects_path = find_in_workspaces(
        search_dirs=['share'],
        project=package_name,
        path=directory,
        first_match_only=True)
    if len(objects_path) == 0:
        raise RuntimeError('Can\'t find directory {}/{}'.format(
            package_name, directory))
    else:
        objects_path = objects_path[0]
    return objects_path

def get_fixed(robot, movable):
    '''Given the robot and movable objects, return all other
    objects in the scene, which are then by definition, the fixed objects'''
    rigid = [body for body in pb_robot.utils.get_bodies() if body.id != robot.id]
    movable_ids = [m.id for m in movable]
    fixed = [body for body in rigid if body.id not in movable_ids]
    return fixed

def ExecuteActions(plan, real=False, pause=True, wait=True, prompt=True, obstacles=[], sim_fatal_failure_prob=0, sim_recoverable_failure_prob=0):
    # if prompt:
    #     input("Execute in Simulation?")
    # obj_held = None
    for name, args in plan:
        # pb_robot.viz.remove_all_debug()
        # bodyNames = [args[i].get_name() for i in range(len(args)) if isinstance(args[i], pb_robot.body.Body)]
        #txt = '{} - {}'.format(name, bodyNames)
        # pb_robot.viz.add_text(txt, position=(0, 0.25, 0.5), size=2)

        executionItems = args[-1]
        for e in executionItems:
            if real:
                e.simulate(timestep=0.1)
            else:
                e.simulate(timestep=0.05)

            # Assign the object being held
            # if isinstance(e, pb_robot.vobj.BodyGrasp):
            #     if name == "pick":
            #         obj_held = e
            #     else:
            #         obj_held = None

            # Simulate failures if specified
            if (name in ["pick", "move_free"] and not isinstance(e, pb_robot.vobj.BodyGrasp)
                and not isinstance(e, pb_robot.vobj.MoveFromTouch)):
                if numpy.random.rand() < sim_fatal_failure_prob:
                    raise ExecutionFailure(fatal=True,
                        reason=f"Simulated fatal failure in {e}")
                elif numpy.random.rand() < sim_recoverable_failure_prob:
                    # if (name in ["place", "place_home", "move_holding"]) or \
                    # (name=="pick" and isinstance(e, pb_robot.vobj.MoveFromTouch)):
                    #     obj_held_arg = obj_held
                    # else:
                    #     obj_held_arg = None
                    raise ExecutionFailure(fatal=False,
                        reason=f"Simulated recoverable failure in {e}")

            if wait:
                input("Next?")
            elif pause:
                time.sleep(0.5)

    if real:
        input("Execute on Robot?")
        try:
            from franka_interface import ArmInterface
        except:
            print("Do not have rospy and franka_interface installed.")
            return

        #try:
        arm = ArmInterface()
        arm.set_joint_position_speed(0.3)
        #except:
        #    print("Unable to connect to real robot. Exiting")
    #           return

        print("Executing on real robot")
        input("start?")
        for name, args in plan:
            executionItems = args[-1]
            for e in executionItems:
                e.execute(realRobot=arm, obstacles=obstacles)
                #input("Next?")

def create_pb_robot_urdf(obj, fname):
    full_urdf_folder = 'pb_robot/tmp_urdfs'
    pb_urdf_folder = 'tmp_urdfs'

    urdf = object_to_urdf(obj)
    path = os.path.join(full_urdf_folder, fname)
    with open(path, 'w') as handle:
        handle.write(str(urdf))

    pb_path = os.path.join(pb_urdf_folder, fname)
    return pb_path

def setup_panda_world(robot, blocks, xy_poses=None, use_platform=True):
    # Adjust robot position such that measurements match real robot reference frame
    robot_pose = numpy.eye(4)
    robot.set_transform(robot_pose)

    pddl_blocks = []

    full_urdf_folder = 'pb_robot/tmp_urdfs'

    if not os.path.exists(full_urdf_folder):
        os.makedirs(full_urdf_folder)

    for block in blocks:
        pb_block_fname = create_pb_robot_urdf(block, block.name + '.urdf')
        pddl_block = pb_robot.body.createBody(pb_block_fname)
        pddl_blocks.append(pddl_block)

    table_x_offset = 0.2
    floor_path = 'tamp/models/panda_table.urdf'
    shutil.copyfile(floor_path, 'pb_robot/models/panda_table.urdf')
    table_file = os.path.join('models', 'panda_table.urdf')
    pddl_table = pb_robot.body.createBody(table_file)
    pddl_table.set_point([table_x_offset, 0, 0])

    frame_path = 'tamp/models/panda_frame.urdf'
    shutil.copyfile(frame_path, 'pb_robot/models/panda_frame.urdf')
    frame_file = os.path.join('models', 'panda_frame.urdf')
    pddl_frame = pb_robot.body.createBody(frame_file)
    pddl_frame.set_point([table_x_offset + 0.762 - 0.0127, 0 + 0.6096 - 0.0127, 0])

    wall_path = 'tamp/models/walls.urdf'
    shutil.copyfile(wall_path, 'pb_robot/models/walls.urdf')
    wall_file = os.path.join('models', 'walls.urdf')
    pddl_wall = pb_robot.body.createBody(wall_file)
    pddl_wall.set_point([table_x_offset + 0.762 + 0.005, 0, 0])

    # Set the initial positions randomly on table.
    if xy_poses is None:
        storage_poses = [(-0.4, -0.45), (-0.4, -0.25), # Left Corner
                         (-0.25, -0.5), (-0.4, 0.25),   # Back Center
                         (-0.4, 0.45), (-0.25, 0.5),   # Right Corner
                         (-0., -0.5), (0., -0.35),   # Left Side
                         (-0., 0.5), (0., 0.35)]     # Right Side
        print('Placing blocks in storage locations...')
        for ix, block in enumerate(pddl_blocks):
            x, y = storage_poses[ix]
            dimensions = numpy.array(block.get_dimensions()).reshape((3, 1))
            if ix < 6 and (ix not in [2, 5]):  # Back storage should have long side along y-axis.
                for rot in all_rotations():
                    rot_dims = numpy.abs(rot.as_matrix()@dimensions)[:, 0]
                    if rot_dims[1] >= rot_dims[0] and rot_dims[1] >= rot_dims[2]:
                        block.set_base_link_pose(((x, y, 0.), rot.as_quat()))
                        break
            else:  # Side storage should have long side along x-axis.
                for rot in all_rotations():
                    rot_dims = numpy.abs(rot.as_matrix()@dimensions)[:, 0]
                    if rot_dims[0] >= rot_dims[1] and rot_dims[0] >= rot_dims[2]:
                        block.set_base_link_pose(((x, y, 0.), rot.as_quat()))
                        break

            z = pb_robot.placements.stable_z(block, pddl_table)
            block.set_base_link_point([x, y, z])
    else:
        for i, (block, xy_pose) in enumerate(zip(pddl_blocks, xy_poses)):

            full_pose = Pose(Position(xy_pose.pos.x,
                                     xy_pose.pos.y,
                                     xy_pose.pos.z),
                            xy_pose.orn)
            block.set_base_link_pose(full_pose)


    # Setup platform.
    if use_platform:
        platform, leg = Object.platform()
        pb_platform_fname = create_pb_robot_urdf(platform, 'platform.urdf')
        pb_leg_fname = create_pb_robot_urdf(leg, 'leg.urdf')
        pddl_platform = pb_robot.body.createBody(pb_platform_fname)
        pddl_leg = pb_robot.body.createBody(pb_leg_fname)

        rotation = pb_robot.geometry.Euler(yaw=numpy.pi/2)
        pddl_platform.set_base_link_pose(pb_robot.geometry.multiply(pb_robot.geometry.Pose(euler=rotation), pddl_platform.get_base_link_pose()))
        pddl_leg.set_base_link_pose(pb_robot.geometry.multiply(pb_robot.geometry.Pose(euler=rotation), pddl_leg.get_base_link_pose()))

        table_z = pddl_table.get_base_link_pose()[0][2]
        pddl_leg.set_base_link_point([0.7, -0.4, table_z + leg.dimensions.z/2])
        pddl_platform.set_base_link_point([0.7, -0.4, table_z + leg.dimensions.z + platform.dimensions.z/2.])
    else:
        pddl_platform = None
        pddl_leg = None

    return pddl_blocks, pddl_platform, pddl_leg, pddl_table, pddl_frame, pddl_wall


def get_pddlstream_info(robot, fixed, movable, add_slanted_grasps, approach_frame, use_vision, home_poses=None):
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


def get_pddl_block_lookup(blocks, pddl_blocks):
    """ Unrotate all blocks and build a map to PDDL. (i.e., use the block.rotation for orn) """
    pddl_block_lookup = {}
    for block in blocks:
        for pddl_block in pddl_blocks:
            if block.name in pddl_block.get_name():
                pddl_block_lookup[block.name] = pddl_block
    return pddl_block_lookup


def print_planning_problem(init, goal, fixed_objs):
    """ Printing function to debug PDDL planning problems """
    print("\n===FIXED OBJECTS===")
    print(fixed_objs)

    print("\n===INITIAL CONDITIONS===")
    for elem in init:
        print_elem = [e for e in elem]
        for i, item in enumerate(print_elem):
            if isinstance(item, pb_robot.vobj.BodyPose):
                pose_list = item.pose[0] + item.pose[1]
                print_elem[i] = "Pose: " + \
                    " ".join("%.2f" % p for p in pose_list)
        print(print_elem)

    print("\n===GOAL CONDITIONS===")
    for elem in goal[1:]:
        print_elem = [e for e in elem]
        for i, item in enumerate(print_elem):
            if isinstance(item, pb_robot.vobj.BodyPose):
                pose_list = item.pose[0] + item.pose[1]
                print_elem[i] = "Pose: " + \
                    " ".join("%.2f" % p for p in pose_list)
        print(print_elem)

    print("\n")
