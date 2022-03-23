from distutils.util import execute
from re import L
import IPython
import numpy as np
import os
import pb_robot
import panda_controls
from pybullet_object_models import ycb_objects
import pybullet as p
import time
import trimesh
from trimesh.viewer import SceneViewer

RED = [255, 0, 0, 255]
GRIPPER_WIDTH = 0.08
ANTIPODAL_TOLERANCE = np.deg2rad(30)

def extract_normal(mesh, index):
    return np.array(mesh.face_normals[index, :])


def sample_antipodal(mesh_fname, body):
    mesh = pb_robot.meshes.read_obj(mesh_fname, decompose=False)
    tmesh = trimesh.Trimesh(mesh.vertices, mesh.faces)
    tmesh.fix_normals()
    # tmesh.show()
    count, total = 0, 0
    points = []
    # Routine for finding an antipodal grasp (rejection sampling).
    while True:
        [point1, point2], [index1, index2] = tmesh.sample(2, return_index=True)
        total += 1
        distance = pb_robot.geometry.get_distance(point1, point2)
        if distance > GRIPPER_WIDTH or distance < 1e-3:
            continue

        direction = point2 - point1
        normal1, normal2 = extract_normal(tmesh, index1), extract_normal(tmesh, index2)
        # Make sure normals are pointing away from each other.
        # if normal1.dot(-direction) < 0:
        #     normal1 *= -1
        # if normal2.dot(direction) < 0:
        #     normal2 *= -1
        error1 = pb_robot.geometry.angle_between(normal1, -direction)
        error2 = pb_robot.geometry.angle_between(normal2, direction)

        # For anitpodal grasps, the angle between the normal and direction vector should be small.
        if (error1 > ANTIPODAL_TOLERANCE) or (error2 > ANTIPODAL_TOLERANCE):
            continue

        count +=1 
        points += [point1, point2]
        break

    print('%d/%d' % (count, total))

    # TODO: Align mesh with URDF in PyBullet.
    mesh_pos = p.getVisualShapeData(body.id)[0][5]
    mesh_orn = p.getVisualShapeData(body.id)[0][6]
    mesh_tform = pb_robot.geometry.tform_from_pose((mesh_pos, mesh_orn))
    object_tform = pb_robot.geometry.tform_from_pose(body.get_base_link_pose())

    for point in points:
        vec = np.array([[point[0], point[1], point[2], 1]]).T
        pb_point = (object_tform@mesh_tform@vec)[0:3, 0]

        visualShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE, rgbaColor=[1, 1, 1, 1], radius=0.005)
        p.createMultiBody(baseMass=0,
                          baseCollisionShapeIndex=-1,
                          baseVisualShapeIndex=visualShapeId,
                          basePosition=pb_point,
                          useMaximalCoordinates=True)

    # TODO: Visualize mesh and points on mesh.
    point_cloud = trimesh.points.PointCloud(points, [RED]*len(points))
    scene = trimesh.scene.Scene([tmesh, point_cloud])
    scene.show()

    # TODO: Test on multiple objects.
    IPython.embed()

    pass


if __name__ == '__main__':
    pb_robot.utils.connect(use_gui=True)
    pb_robot.utils.set_default_camera()

    floor_file = 'models/short_floor.urdf'
    floor = pb_robot.body.createBody(floor_file)

    mustard = pb_robot.body.createBody(os.path.join(ycb_objects.getDataPath(), 'YcbMustardBottle', 'model.urdf'))
    mustard.set_base_link_pose(((0, 0, pb_robot.placements.stable_z(mustard, floor)), (0, 0, 0, 1)))

    drill = pb_robot.body.createBody(os.path.join(ycb_objects.getDataPath(), 'YcbPowerDrill', 'model.urdf'))
    drill.set_base_link_pose(((0.25, 0, pb_robot.placements.stable_z(drill, floor)), (0, 0, 0, 1)))

    sample_antipodal(os.path.join(ycb_objects.getDataPath(), 'YcbPowerDrill', 'textured_simple_reoriented.obj'), drill)
    IPython.embed()
    pb_robot.utils.wait_for_user()
    pb_robot.utils.disconnect()

