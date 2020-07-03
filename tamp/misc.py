import numpy
import pb_robot
import os
import shutil

from block_utils import object_to_urdf


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

def ExecuteActions(manip, plan):
    for name, args in plan:
        pb_robot.viz.remove_all_debug()
        bodyNames = [args[i].get_name() for i in range(len(args)) if isinstance(args[i], pb_robot.body.Body)]
        txt = '{} - {}'.format(name, bodyNames)
        pb_robot.viz.add_text(txt, position=(0, 0.25, 0.5), size=2)

        executionItems = args[-1]
        for e in executionItems:
            e.simulate()
            input("Next?")

def ComputePrePose(og_pose, directionVector, relation=None):
    backup = numpy.eye(4)
    backup[0:3, 3] = directionVector
    prepose = numpy.dot(og_pose, backup)
    if relation is not None:
        prepose = numpy.dot(prepose, relation)
    return prepose

def setup_panda_world(robot, blocks):
    # Adjust robot position such that measurements match real robot reference frame
    robot_pose = numpy.eye(4)
    robot_pose[2, 3] -= 0.1
    robot.set_transform(robot_pose)

    pddl_blocks = []

    full_urdf_folder = 'pb_robot/tmp_urdfs'
    pb_urdf_folder = 'tmp_urdfs'

    if not os.path.exists(full_urdf_folder):
        os.makedirs(full_urdf_folder)

    for block in blocks:
        block_urdf = object_to_urdf(block)
        block_fname = os.path.join(full_urdf_folder, str(block)+'.urdf')
        with open(block_fname, 'w') as handle:
            handle.write(str(block_urdf))

        pb_block_fname = os.path.join(pb_urdf_folder, str(block)+'.urdf')
        pddl_block = pb_robot.body.createBody(pb_block_fname)
        pddl_blocks.append(pddl_block)
    
    floor_path = 'tamp/models/short_floor.urdf'
    shutil.copyfile(floor_path, 'pb_robot/models/short_floor.urdf')
    table_file = os.path.join('models', 'short_floor.urdf')
    table = pb_robot.body.createBody(table_file)
    table.set_point([0.2, 0, -0.11])

    # Set the initial positions randomly.
    table_top_z = -0.11 + 1e-5
    for block in pddl_blocks:
        block.set_base_link_point([0.6, 0.0, table_top_z + 0.02])
    
    return pddl_blocks
