import csv
import numpy as np
import odio_urdf
import os
import pybullet as p
import shutil
import time

from collections import namedtuple
from copy import copy
from datetime import datetime
from scipy.spatial.transform import Rotation as R

from pybullet_utils import PyBulletServer, quat_math


ParticleDistribution = namedtuple('ParticleDistribution', 'particles weights')
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

ZERO_ROT = Quaternion(0, 0, 0, 1)
ZERO_POS = Position(0, 0, 0)
ZERO_POSE = Pose(ZERO_POS, ZERO_ROT)


class Object:

    def __init__(self, name, dimensions, mass, com, color):
        self.name = name
        self.dimensions = dimensions    # Dimensions
        self.mass = mass                # float
        self.com = com                  # Position, position of COM relative to
                                        # center of object
        self.color = color              # Color
        self.pose = ZERO_POSE           # Pose (set later)
        self.id = None                  # int (set later)
        self.com_filter = None          # ParticleDistribution (set later)

    def set_pose(self, pose):
        self.pose = pose

    def get_pose(self):
        return self.pose

    def set_id(self, id):
        self.id = id

    def get_id(self):
        return self.id

    def vectorize(self):
      """ Summarize a block in a vector

      1 mass
      3 com
      3 dimensions
      3 pos
      4 quat
      3 color
      """
      v = np.zeros(17)
      v[0] = self.mass
      v[1:4] = self.com
      v[4:7] = self.dimensions
      v[7:10] = self.pose.pos
      v[10:14] = self.pose.orn
      v[14:17] = self.color

      return v

    @staticmethod
    def random(name=None):
        """ Construct a random object

        Arguments:
            name {str} -- name of the object

        Returns:
            Object -- a random object
        """
        # blocks range in size from 0.1 to 1
        dims = Dimensions(*(np.random.rand(3) * 0.1 + 0.05))
        # pick a density and multiply by the volume to get mass
        density = np.random.rand() * 0.9 + 0.1
        mass = np.random.uniform(0.1, 1.0)
        # center of mass lies within the middle 0.9 of the block along each axis
        com = Position(*((np.random.rand(3) - 0.5) * 0.9 * dims))
        # pick a random color
        color = Color(*np.random.rand(3))
        # and add the new block to the list
        return Object(name, dims, mass, com, color)

    @staticmethod
    def platform():
        leg_height = 0.15

        platform_block = Object(name='platform',
                      dimensions=Dimensions(x=0.3, y=0.2, z=0.01),  # z=0.15
                      mass=0,  # 100.
                      com=Position(x=0., y=0., z=0.),
                      color=Color(r=0.25, g=0.25, b=0.25))
        platform_block.set_pose(Pose(pos=Position(x=0., y=0., z=leg_height + 0.005),
                                orn=Quaternion(x=0, y=0, z=0, w=1)))

        platform_leg = Object(name='platform-leg',
                      dimensions=Dimensions(x=0.05, y=0.05, z=leg_height),  # z=0.15
                      mass=0,  
                      com=Position(x=0., y=0., z=0.),
                      color=Color(r=0.25, g=0.25, b=0.25))
        platform_leg.set_pose(Pose(pos=Position(x=0., y=0., z=leg_height/2.),
                                orn=Quaternion(x=0, y=0, z=0, w=1)))

        return platform_block, platform_leg

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

    def set_pos(self, new_pos, client):
        self.pos = new_pos
        if self.c_id == -1:
           self.c_id = p.createConstraint(parentBodyUniqueId=self.hand_id,
                                          parentLinkIndex=-1,
                                          childBodyUniqueId=-1,
                                          childLinkIndex=-1,
                                          jointType=p.JOINT_FIXED,
                                          jointAxis=(0,0,0),
                                          parentFramePosition=(0,0,0),
                                          childFramePosition=new_pos,
                                          physicsClientId=client)
        else:
            p.changeConstraint(userConstraintUniqueId=self.c_id,
                               jointChildPivot=new_pos,
                               physicsClientId=client)

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

    def set_hand_pos(self, pos, client):
        global_pos = Position(*np.add(pos, [self.offset[0], self.offset[1], 0]))
        self.hand.set_pos(global_pos, client)

    def get_hand_pos(self):
        global_pos = self.hand.get_pos()
        rel_pos = Position(*np.subtract(global_pos, [self.offset[0], self.offset[1], 0]))
        return rel_pos

class Environment:

    def __init__(self, worlds, vis_sim=True, vis_frames=False, use_hand=True, save_tower=False):
        self.vis_sim = vis_sim
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
                        hand_pose = Position(x=x_pos, y=y_pos-0.25, z=2.25)
                        hand_id = self.pybullet_server.load_urdf(self.tmp_dir+'/hand_'+str(world_i)+'.urdf',
                                                            hand_pose)
                        self.worlds[world_i].set_hand_id(hand_id)
                        self.worlds[world_i].set_hand_pos(Position(x=0, y=-0.25, z=2.25),
                                                          self.pybullet_server.client)

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
                            # NOTE(izzy): I haven't tested this yet!
                            #self.pybullet_server.vis_particles(obj)
                    world_i += 1

        if save_tower:
            # save urdfs
            timestamp = datetime.now().strftime("%d-%m-%H-%M-%S")
            tower_dir = 'tower-'+timestamp
            os.mkdir(tower_dir)
            urdfs_dir = os.path.join(tower_dir, 'urdfs')
            os.mkdir(urdfs_dir)
            for obj in self.worlds[0].objects:
                shutil.copyfile(os.path.join(self.tmp_dir, str(obj)+'.urdf'),
                                os.path.join(urdfs_dir, str(obj)+'.urdf'))

            # save list of urdf_names and poses to csv file
            filepath = tower_dir+'/obj_poses.csv'
            with open(filepath, 'w') as handle:
                obj_writer = csv.writer(handle)
                for obj in self.worlds[0].objects:
                    pos, orn = self.pybullet_server.get_pose(obj.id)
                    row = [str(obj)+'.urdf']+\
                            [str(p) for p in pos]+\
                            [str(o) for o in orn]+\
                            [str(d) for d in obj.dimensions]
                    obj_writer.writerow(row)
            print('Saved tower URDFs and pose .csv to: ', tower_dir)

    def step(self, action=None, vis_frames=False):
        # Apply every action.
        if action and action.__class__.__name__ == 'PushAction':
            hand_pos = action.step()
            for world in self.worlds:
                world.set_hand_pos(hand_pos)

        # forward step the sim
        self.pybullet_server.step()

        # update all world object poses
        for world in self.worlds:
            for obj in world.objects:
                pos, orn = self.pybullet_server.get_pose(obj.id)
                dynamics = p.getDynamicsInfo(bodyUniqueId=obj.id,
                                             linkIndex=-1,
                                             physicsClientId=self.pybullet_server.client)
                (point, quat) = p.multiplyTransforms(pos,
                                                     orn,
                                                     *p.invertTransform(dynamics[3],
                                                     dynamics[4]))
                # pyBullet returns the pose of the COM, not COG
                obj.set_pose(Pose(Position(*point), Quaternion(*quat)))
                if vis_frames:
                    #pos, quat = obj.get_pose()
                    self.pybullet_server.vis_frame(pos, orn)

        # sleep (for visualization purposes)
        if self.vis_sim:
            time.sleep(0.005)

    def disconnect(self):
        self.pybullet_server.disconnect()

    def cleanup(self):
        # remove temp urdf files (they will accumulate quickly)
        shutil.rmtree(self.tmp_dir)

def add_noise(pose, cov):
        pos = Position(*np.random.multivariate_normal(mean=pose.pos, cov=cov))
        orn = pose.orn
        return Pose(pos, orn)

# see if the two dicts of positions are roughly equivalent
def pos_unchanged(init_poses, final_poses, eps=2e-3):
    dists = [np.linalg.norm(np.array(init_poses[obj].pos)
            - np.array(final_poses[obj])) for obj in init_poses]
    print('Dists', dists)
    return max(dists) < eps

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
    I = 0.001
    link_urdf = odio_urdf.Link(object.name,
                  odio_urdf.Inertial(
                      odio_urdf.Origin(xyz=tuple(object.com), rpy=(0, 0, 0)),
                      odio_urdf.Mass(value=object.mass),
                      odio_urdf.Inertia(ixx=I,
                                        ixy=0,
                                        ixz=0,
                                        iyy=I,
                                        iyz=0,
                                        izz=I)
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

    object_urdf = odio_urdf.Robot(link_urdf, name=object.name)
    return object_urdf

# list of length 3 of (min, max) ranges for each dimension
def get_com_ranges(object):
    half_dims = np.array(object.dimensions) * 0.5
    return np.array([-half_dims, half_dims]).T

def group_blocks(bottom, top):
    total_mass = bottom.mass + top.mass
    # we'll use the pos from the bottom block as the center of geometry
    new_pos = bottom.pose.pos
    new_dims = bottom.dimensions
    # take a weighted average of the COM vectors
    bottom_vec = np.array(bottom.com) + np.array(bottom.pose.pos)
    bottom_frac = bottom.mass / total_mass
    top_vec = np.array(top.com) + np.array(top.pose.pos)
    top_frac = top.mass / total_mass
    # bring the COM vector back into the coordinate frame of the group
    # by subtracting new_pos
    new_com = Position(*(bottom_vec*bottom_frac + top_vec*top_frac - new_pos))
    # construct a new block with the attributes of the group
    new_block = Object('group', None, total_mass, new_com, None)
    new_block.pose = Pose(new_pos, ZERO_ROT)
    new_block.dimensions = new_dims

    return new_block

def get_rotated_block(block):
    """ Take a block which is rotated by an element of the rotation group of a
    cube, and produce a new block with no rotation, but with changed COM and
    dimensions such that it is equivalent to the previous block

    Arguments:
        block {Object} -- the original block

    Returns:
        Object -- the rotated block
    """
    # create a new block
    new_block = copy(block)
    # set it to to have the same pos with no rotation
    new_pose = Pose(block.pose.pos, Quaternion(0,0,0,1))
    new_block.set_pose(new_pose)
    # get the original block's rotation
    r = R.from_quat(block.pose.orn)
    # rotate the old center of mass
    new_block.com = Position(*r.apply(block.com))
    # rotate the old dimensions
    new_block.dimensions = Dimensions(*np.abs(r.apply(block.dimensions)))
    # rotate the particle filter for the com if there is one
    if block.com_filter is not None:
        new_block.com_filter = ParticleDistribution(
            r.apply(block.com_filter.particles), block.com_filter.weights)

    new_block.rotation = block.pose.orn

    return new_block

def rotation_group():
    V = np.eye(3) * np.pi/2
    for v in V:
        for r in [0, np.pi/2]:
            v[0] = r
            yield R.from_euler('zyx', v)

def get_adversarial_blocks(num_blocks=4):
    b1 = Object(name='block1',
                dimensions=Dimensions(0.02, 0.1, 0.02),
                mass=1.,
                com=Position(0.0075, 0.0475, 0),
                color=Color(0, 0, 1))
    b2 = Object(name='block2',
                dimensions=Dimensions(0.02, 0.1, 0.02),
                mass=1.,
                com=Position(-0.0075, -0.0475, 0),
                color=Color(1, 0, 1))
    b3 = Object(name='block3',
                dimensions=Dimensions(0.04, 0.12, 0.04),
                mass=1.,
                com=Position(0, 0.05, 0.),
                color=Color(0, 1, 1))
    b4 = Object(name='block4',
                dimensions=Dimensions(0.12, 0.02, 0.04),
                mass=1.,
                com=Position(-0.0575, 0, -0.008),
                color=Color(0, 1, 0))
    return [b1, b2, b3, b4][:num_blocks]


if __name__ == '__main__':
    blocks = get_adversarial_blocks()

    for r in rotation_group():
        b = copy(blocks[0])
        b.pose = Pose(ZERO_POS, Quaternion(*r.as_quat()))
        b = get_rotated_block(b)

        print(b.com, b.dimensions)