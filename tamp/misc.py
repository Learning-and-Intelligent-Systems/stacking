import numpy
import pb_robot
import os
import shutil
import time

import tamp.primitives as primitives

from block_utils import object_to_urdf, Object
from pddlstream.language.generator import from_gen_fn, from_list_fn, from_fn, from_test , BoundedGenerator
from pddlstream.utils import read

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

def ExecuteActions(manip, plan, pause=True):
    for name, args in plan:
        pb_robot.viz.remove_all_debug()
        bodyNames = [args[i].get_name() for i in range(len(args)) if isinstance(args[i], pb_robot.body.Body)]
        txt = '{} - {}'.format(name, bodyNames)
        pb_robot.viz.add_text(txt, position=(0, 0.25, 0.5), size=2)

        executionItems = args[-1]
        for e in executionItems:
            e.simulate()
            if pause:
                input("Next?")
            else:
                time.sleep(0.5)

def ComputePrePose(og_pose, directionVector, relation=None):
    backup = numpy.eye(4)
    backup[0:3, 3] = directionVector
    prepose = numpy.dot(og_pose, backup)
    if relation is not None:
        prepose = numpy.dot(prepose, relation)
    return prepose

def create_pb_robot_urdf(obj, fname):
    full_urdf_folder = 'pb_robot/tmp_urdfs'
    pb_urdf_folder = 'tmp_urdfs'

    urdf = object_to_urdf(obj)
    path = os.path.join(full_urdf_folder, fname)
    with open(path, 'w') as handle:
        handle.write(str(urdf))

    pb_path = os.path.join(pb_urdf_folder, fname)
    return pb_path

def setup_panda_world(robot, blocks):
    # Adjust robot position such that measurements match real robot reference frame
    robot_pose = numpy.eye(4)
    robot_pose[2, 3] -= 0.1
    robot.set_transform(robot_pose)

    pddl_blocks = []

    full_urdf_folder = 'pb_robot/tmp_urdfs'

    if not os.path.exists(full_urdf_folder):
        os.makedirs(full_urdf_folder)

    for block in blocks:
        pb_block_fname = create_pb_robot_urdf(block, block.name + '.urdf')
        pddl_block = pb_robot.body.createBody(pb_block_fname)
        pddl_blocks.append(pddl_block)
    
    floor_path = 'tamp/models/short_floor.urdf'
    shutil.copyfile(floor_path, 'pb_robot/models/short_floor.urdf')
    table_file = os.path.join('models', 'short_floor.urdf')
    pddl_table = pb_robot.body.createBody(table_file)
    pddl_table.set_point([0.2, 0, -0.11])

    # Set the initial positions randomly on table.
    for ix, block in enumerate(pddl_blocks):
        while True:
            z = pb_robot.placements.stable_z(block, pddl_table)
            x = numpy.random.uniform(0.4, 0.7)
            y = numpy.random.uniform(0.1, 0.4)
            block.set_base_link_point([x, y, z])

            # Check that there is no collision with already placed blocks.
            collision = False
            for jx in range(0, ix):
                if pb_robot.collisions.body_collision(block, pddl_blocks[jx], max_distance=0.1):
                    collision = True
            if collision:
                continue
            break

    # Setup platform.
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


    return pddl_blocks, pddl_platform, pddl_leg, pddl_table


def get_pddlstream_info(robot, fixed, movable):
    domain_pddl = read('tamp/domain_stacking.pddl') 
    stream_pddl = read('tamp/stream_stacking.pddl') 
    constant_map = {}
    
    stream_map = {
        'sample-pose-table': from_gen_fn(primitives.get_stable_gen_table(fixed)),
        'sample-pose-block': from_fn(primitives.get_stable_gen_block(fixed)),
        'sample-grasp': from_gen_fn(primitives.get_grasp_gen(robot)),
        'inverse-kinematics': from_fn(primitives.get_ik_fn(robot, fixed)), 
        'plan-free-motion': from_fn(primitives.get_free_motion_gen(robot, fixed)),
        'plan-holding-motion': from_fn(primitives.get_holding_motion_gen(robot, fixed)),
    }

    return domain_pddl, constant_map, stream_pddl, stream_map
