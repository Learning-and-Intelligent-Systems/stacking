import time
import numpy as np
import os
from collections import namedtuple
from copy import copy

import pybullet as p
import odio_urdf
from pybullet_utils import PyBulletServer, quat_math

Position = namedtuple('Position', 'x y z')
Quaternion = namedtuple('Quaternion', 'x y z w')
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
Contact = namedtuple('Contact', 'objectA_name objectB_name pose_a_b')
'''
:param objectA_name: string, name of object A involved in contact
:param objectB_name: string, name of object B involved in contact
:param pose_a_b: Pose, the relative pose of object A's CENTER (OF GEOMETRY, NOT COM)
                in object B's center
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
        self.id = None                  # int (set later)

    def set_pose(self, pose):
        self.pose = pose

    def get_pose(self):
        return self.pose

    def set_id(self, id):
        self.id = id

    def get_id(self):
        return self.id

    def random(name):
        """ Construct a random object


        Arguments:
            name {str} -- name of the object

        Returns:
            Object -- a random object
        """
        # blocks range in size from 0.1 to 1
        dims = Dimensions(*(np.random.rand(3) * 0.9 + 0.1))
        # pick a density and multiply by the volume to get mass
        density = np.random.rand() * 0.9 + 0.1
        mass = density * dims.x * dims.y * dims.z
        # center of mass lies within the middle 0.75 of the block along each axis
        com = Position(*((np.random.rand(3) - 0.5) * 0.75 * dims))
        # pick a random color
        color = Color(*np.random.rand(3))
        # and add the new block to the list
        return Object(name, dims, mass, com, color)


class Hand:
    def __init__(self):
        """ Note that Hand will store the global position of the hand (as directly
            returned by PyBullet. To get the position of the hand relative to a
            single world, use the world object.
        """
        self.hand_id = -1
        self.c_id = -1
        self.pos = None

    def get_pos(self):
        return self.pos

    def set_pos(self, new_pos):
        self.pos = new_pos
        if self.c_id == -1:
           self.c_id = p.createConstraint(parentBodyUniqueId=self.hand_id,
                                          parentLinkIndex=-1,
                                          childBodyUniqueId=-1,
                                          childLinkIndex=-1,
                                          jointType=p.JOINT_FIXED,
                                          jointAxis=(0,0,0),
                                          parentFramePosition=(0,0,0),
                                          childFramePosition=new_pos)
        else:
            p.changeConstraint(userConstraintUniqueId=self.c_id,
                               jointChildPivot=new_pos)

    def set_id(self, hand_id):
        self.hand_id = hand_id


class World:

    def __init__(self, objects):
        self.objects = objects
        self.offset = None          # offset in the env (set later)
        self.hand = Hand()

    def get_poses(self):
        return {w_obj.name: w_obj.get_pose().pos
            for w_obj in self.objects}

    def get_pose(self, obj):
        for w_obj in self.objects:
            if w_obj == obj:
                global_pose = obj.get_pose()
                world_pos = Position(*np.subtract(global_pose.pos,
                                                    [self.offset[0], self.offset[1], 0.0]))
                world_pose = Pose(world_pos, global_pose.orn)
                return world_pose

    def set_offset(self, offset):
        self.offset = offset
        for object in self.objects:
            offset_pos = Position(*np.add(object.pose.pos,
                                    [self.offset[0], self.offset[1], 0.0]))
            orn = object.pose.orn
            object.set_pose(Pose(offset_pos, orn))

    def set_hand_id(self, h_id):
        self.hand.set_id(h_id)

    def set_hand_pos(self, pos):
        global_pos = Position(*np.add(pos, [self.offset[0], self.offset[1], 0]))
        self.hand.set_pos(global_pos)

    def get_hand_pos(self):
        global_pos = self.hand.get_pos()
        rel_pos = Position(*np.subtract(global_pos, [self.offset[0], self.offset[1], 0]))
        return rel_pos

class Environment:

    def __init__(self, worlds, vis_sim=True, vis_frames=False, use_hand=True):
        self.worlds = worlds
        self.pybullet_server = PyBulletServer(vis_sim)

        # load ground plane
        self.plane_id = self.pybullet_server.load_urdf("plane_files/plane.urdf", \
                            Position(0.,0.,0.))

        # make dir to hold temp urdf files
        self.tmp_dir = 'tmp_urdfs'
        if not os.path.isdir(self.tmp_dir):
            os.mkdir(self.tmp_dir)

        # load objects from each world and set object link ids
        sqrt_nworlds = int(np.ceil(np.sqrt(len(self.worlds))))
        spacing = 1.0
        world_i = 0
        for x_pos in np.linspace(-spacing*(sqrt_nworlds-1)/2, spacing*(sqrt_nworlds-1)/2, sqrt_nworlds):
            for y_pos in np.linspace(-spacing*(sqrt_nworlds-1)/2, spacing*(sqrt_nworlds-1)/2, sqrt_nworlds):
                if world_i < len(self.worlds):
                    world_center = (x_pos, y_pos)
                    worlds[world_i].set_offset(world_center)

                    # Load the gripper object.
                    if use_hand:
                        with open(self.tmp_dir+'/hand_'+ str(world_i) + '.urdf', 'w') as handle:
                            handle.write(str(hand_urdf()))
                        hand_pose = Position(x=x_pos, y=y_pos-0.25, z=0.25)
                        hand_id = self.pybullet_server.load_urdf(self.tmp_dir+'/hand_'+str(world_i)+'.urdf',
                                                            hand_pose)
                        self.worlds[world_i].set_hand_id(hand_id)
                        self.worlds[world_i].set_hand_pos(Position(x=0, y=-0.25, z=0.25))

                    for obj in self.worlds[world_i].objects:
                        object_urdf = object_to_urdf(obj)

                        with open(self.tmp_dir+'/'+str(obj)+'.urdf', 'w') as handle:
                            handle.write(str(object_urdf))
                        # I think there is a bug in this pyBullet function. The documentation
                        # says the position should be of the inertial frame, but it only
                        # works if you give it the position of the center of geometry, not
                        # the center of mass/inertial frame
                        obj_id = self.pybullet_server.load_urdf(self.tmp_dir+'/'+str(obj)+'.urdf',
                                                                obj.pose.pos,
                                                                obj.pose.orn)
                        obj.set_id(obj_id)
                        if vis_frames:
                            pos, quat = self.pybullet_server.get_pose(obj_id)
                            self.pybullet_server.vis_frame(pos, quat)
                    world_i += 1

    def step(self, action=None, vis_frames=False):
        # Apply every action.
        if action:
            hand_pos = action.step()
            for world in self.worlds:
                world.set_hand_pos(hand_pos)

        # forward step the sim
        self.pybullet_server.step()

        # update all world object poses
        for world in self.worlds:
            for obj in world.objects:
                pos, orn = self.pybullet_server.get_pose(obj.id)
                obj.set_pose(Pose(Position(*pos), Quaternion(*orn)))
                if vis_frames:
                    pos, quat = obj.get_pose()
                    self.pybullet_server.vis_frame(pos, orn)

        # sleep (for visualization purposes)
        time.sleep(0.05)

    def disconnect(self):
        self.pybullet_server.disconnect()

    def cleanup(self):
        # remove temp urdf files (they will accumulate quickly)
        shutil.rmtree(self.tmp_dir)

def simulate_from_contacts(objects, contacts, vis=True, T=60):
    world = World(objects.values())
    init_poses = get_poses_from_contacts(contacts)
    # set object poses in all worlds
    for (obj, pose) in init_poses.items():
        for world_obj in world.objects:
            if world_obj.name == obj:
                world_obj.set_pose(pose)

    env = Environment([world], vis_sim=vis, use_hand=False)
    for t in range(T):
        env.step(vis_frames=vis)
    env.disconnect()

    return world.get_poses()

def hand_urdf():
    rgb = (0, 1, 0)
    link_urdf = odio_urdf.Link('hand',
                    odio_urdf.Inertial(
                      odio_urdf.Origin(xyz=(0, 0, 0), rpy=(0, 0, 0)),
                      odio_urdf.Mass(value=0.1),
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
                          odio_urdf.Sphere(radius=0.02)
                      )
                  ),
                  odio_urdf.Visual(
                      odio_urdf.Origin(xyz=(0, 0, 0), rpy=(0, 0, 0)),
                      odio_urdf.Geometry(
                          odio_urdf.Sphere(radius=0.02),
                      ),
                      odio_urdf.Material('color',
                                    odio_urdf.Color(rgba=(*rgb, 1.0))
                                    )
                  ))

    object_urdf = odio_urdf.Robot(link_urdf)
    return object_urdf

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

# get positions (center of geometry, not COM) from contact state
def get_poses_from_contacts(contacts):
    obj_poses = {'ground': Pose(Position(0.,0.,0.), Quaternion(0., 0., 0., 1.))}
    copy_contacts = copy(contacts)
    while len(copy_contacts) > 0:
        for contact in copy_contacts:
            if contact.objectB_name in obj_poses:
                objA_pos = Position(*np.add(obj_poses[contact.objectB_name].pos,
                                    contact.pose_a_b.pos))
                objA_orn = Quaternion(*quat_math(obj_poses[contact.objectB_name].orn,
                                                contact.pose_a_b.orn))
                obj_poses[contact.objectA_name] = Pose(objA_pos, objA_orn)
                copy_contacts.remove(contact)

    return obj_poses

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
