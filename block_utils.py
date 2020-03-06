import time
import numpy as np
from collections import namedtuple
from copy import copy

import odio_urdf
from pybullet_utils import PyBulletServer

Position = namedtuple('Position', 'x y z')
'''
:param x: float, x position
:param y: float, y position
:param z: float, z position
'''
Dimensions = namedtuple('Dimensions', 'width length height')
'''
:param width: float, width of object (in the x direction)
:param length: float, length of object (in the y direction)
:param height: float, height of object (in the z direction)
'''
Color = namedtuple('Color', 'r g b')
'''
:param r: float in [0.,1.], red value
:param g: float in [0.,1.], green value
:param b: float in [0.,1.], blue value
'''
Object = namedtuple('Object', 'dimensions mass com color')
'''
:param dimensions: Dimensions, dimensions of object
:param mass: float, mass of the object
:param com: Position, position of the COM in the link frame (which is located at the center of the object)
:param color: Color, RGB value of block
'''
Contact = namedtuple('Contact', 'objectA_name objectB_name p_a_b')
'''
:param objectA_name: string, name of object A involved in contact
:param objectB_name: string, name of object B involved in contact
:param p_a_b: Position, the position of object A's CENTER (OF GEOMETRY, NOT COM)
                object B's center
'''

def object_to_urdf(object_name, object):
    rgb = np.random.uniform(0, 1, 3)
    link_urdf = odio_urdf.Link(object_name,
                  odio_urdf.Inertial(
                      odio_urdf.Origin(xyz=tuple(object.com), rpy=(0, 0, 0)),
                      odio_urdf.Mass(value=object.mass),
                      odio_urdf.Inertia(ixx=0.001,
                                        ixy=0,
                                        ixz=0,
                                        iyy=0.001,
                                        iyz=0,
                                        izz=0.001)
                  ),
                  odio_urdf.Collision(
                      odio_urdf.Origin(xyz=(0, 0, 0), rpy=(0, 0, 0)),
                      odio_urdf.Geometry(
                          odio_urdf.Box(size=(object.dimensions.width,
                                                object.dimensions.length,
                                                object.dimensions.height))
                      )
                  ),
                  odio_urdf.Visual(
                      odio_urdf.Origin(xyz=(0, 0, 0), rpy=(0, 0, 0)),
                      odio_urdf.Geometry(
                          odio_urdf.Box(size=(object.dimensions.width,
                                                object.dimensions.length,
                                                object.dimensions.height))
                      ),
                      odio_urdf.Material('color',
                                    odio_urdf.Color(rgba=(*object.color, 1.0))
                                    )
                  ))

    object_urdf = odio_urdf.Robot(link_urdf)
    return object_urdf

def render_objects(objects, obj_ps, steps=500, vis=False, vis_frames=False, cameraDistance=0.4):
    pybullet_server = PyBulletServer(vis, cameraDistance)
    object_models = []
    for obj in obj_ps:
        if obj == 'ground':
            plane_id = pybullet_server.load_urdf("plane_files/plane.urdf", obj_ps[obj])
            object_models.append((obj, plane_id))
        else:
            object_urdf = object_to_urdf(obj, objects[obj])
            with open(obj+'.urdf', 'w') as handle:
                handle.write(str(object_urdf))
            # I think there is a bug in this pyBullet function. The documentation
            # says the position should be of the inertial frame, but it only
            # works if you give it the position of the center of geometry, not
            # the center of mass/inertial frame
            obj_model = pybullet_server.load_urdf(obj+'.urdf', obj_ps[obj])
            object_models.append((obj, obj_model))
            if vis_frames:
                pos, quat = pybullet_server.get_pose(obj_model)
                pybullet_server.vis_frame(pos, quat, lifeTime=steps)

    poses = []
    for t in range(steps):
        poses_t = {}
        pybullet_server.step()
        for obj, obj_model in object_models:
            poses_t[obj] = Position(*pybullet_server.get_pose(obj_model)[0])
        time.sleep(.05)
        poses.append(poses_t)

    pybullet_server.disconnect()
    return poses

# get positions (center of geometry, not COM) from contact state
def get_ps_from_contacts(contacts):
    obj_cog_ps = {'ground': Position(0.,0.,0.)}
    copy_contacts = copy(contacts)
    while len(copy_contacts) > 0:
        for contact in copy_contacts:
            if contact.objectB_name in obj_cog_ps:
                obj_cog_ps[contact.objectA_name] = Position(*np.add(obj_cog_ps[contact.objectB_name], contact.p_a_b))
                copy_contacts.remove(contact)

    return obj_cog_ps

def object_names_in_order(contacts):
    contact_dict = {}
    for contact in contacts:
        contact_dict[contact.objectB_name] = contact.objectA_name

    object_names = ['ground']
    current_object = 'ground'
    for _ in range(len(contacts)):
        current_object = contact_dict[current_object]
        object_names.append(current_object)

    return object_names
