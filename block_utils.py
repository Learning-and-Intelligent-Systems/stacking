import time
import numpy as np
from collections import namedtuple
from copy import copy

import odio_urdf
from pybullet_utils import PyBulletServer

Position = namedtuple('Position', 'x y z')
Orientation = namedtuple('Orientation', 'x y z w')
Pose = namedtuple('Pose', 'pos orn')

Dimensions = namedtuple('Dimensions', 'x y z')
'''
:param x: float, length of object in x direction of object frame
:param y: float, length of object in y direction of object frame
:param z: float, length of object in z direction of object frame
'''
Color = namedtuple('Color', 'r g b')
'''
:param r: float in [0.,1.], red value
:param g: float in [0.,1.], green value
:param b: float in [0.,1.], blue value
'''
Contact = namedtuple('Contact', 'objectA_name objectB_name p_a_b')
'''
:param objectA_name: string, name of object A involved in contact
:param objectB_name: string, name of object B involved in contact
:param p_a_b: Position, the position of object A's CENTER (OF GEOMETRY, NOT COM)
                object B's center
'''

class Object:

    def __init__(self, name, dimensions, mass, com, color):
        self.name = name
        self.dimensions = dimensions    # Dimensions
        self.mass = mass                # float
        self.com = com                  # Position, position of COM relative to
                                        # center of object
        self.color = color              # Color
        self.pose = None                # Pose (set later)

class World:

    def __init__(self, objects):
        self.objects = objects
        self.cs = None              # Contact (set later)

def object_to_urdf(object):
    rgb = np.random.uniform(0, 1, 3)
    link_urdf = odio_urdf.Link(object.name,
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
                          odio_urdf.Box(size=(object.dimensions.x,
                                                object.dimensions.y,
                                                object.dimensions.z))
                      )
                  ),
                  odio_urdf.Visual(
                      odio_urdf.Origin(xyz=(0, 0, 0), rpy=(0, 0, 0)),
                      odio_urdf.Geometry(
                          odio_urdf.Box(size=(object.dimensions.x,
                                                object.dimensions.y,
                                                object.dimensions.z))
                      ),
                      odio_urdf.Material('color',
                                    odio_urdf.Color(rgba=(*object.color, 1.0))
                                    )
                  ))

    object_urdf = odio_urdf.Robot(link_urdf)
    return object_urdf

def render_worlds(worlds, obj_ps, steps=500, vis=False, vis_frames=False, cameraDistance=0.4):
    pybullet_server = PyBulletServer(vis, cameraDistance)
    object_models = []
    world = worlds[0]

    plane_id = pybullet_server.load_urdf("plane_files/plane.urdf", obj_ps['ground'])
    object_models.append(('ground', plane_id))
    for obj in world.objects:
        object_urdf = object_to_urdf(obj)
        with open(obj.name+'.urdf', 'w') as handle:
            handle.write(str(object_urdf))
        # I think there is a bug in this pyBullet function. The documentation
        # says the position should be of the inertial frame, but it only
        # works if you give it the position of the center of geometry, not
        # the center of mass/inertial frame
        obj_model = pybullet_server.load_urdf(obj.name+'.urdf', obj_ps[obj.name])
        object_models.append((obj.name, obj_model))
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

# list of length 3 of (min, max) ranges for each dimension
def get_com_ranges(object):
    half_dims = np.array(object.dimensions) * 0.5
    return np.array([-half_dims, half_dims]).T

# throw away contact geometry. Return dict of pairwise relations between
# objects. By convention, we assume object A is on top of object B
def get_contact_dict(contacts, bottom_up=True):
    contact_dict = {}
    for contact in contacts:
        if bottom_up:
            contact_dict[contact.objectB_name] = contact.objectA_name
        else:
            contact_dict[contact.objectA_name] = contact.objectB_name

    return contact_dict

# follow a list of contacts up from the ground to construct a list of object
# names that describe a tower
def object_names_in_order(contacts):
    contact_dict = get_contact_dict(contacts)
    object_names = ['ground']
    current_object = 'ground'

    for _ in range(len(contacts)):
        current_object = contact_dict[current_object]
        object_names.append(current_object)

    return object_names
