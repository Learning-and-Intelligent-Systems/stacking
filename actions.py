from block_utils import Environment, World, Object, Position, Pose, \
                        Quaternion, Dimensions, Color, get_com_ranges
from filter_utils import create_uniform_particles
import pybullet as p
import copy
import numpy as np


class PushAction:
    def __init__(self, world, block, direction, timesteps, delta=0.005):
        """ PushAction moves the hand in the given world by a fixed distance 
            every timestep. We assume we will push the block in a direction through
            the object's geometric center.
        :param world: The world which this action should apply to. Used to get the hand
                      and calculate offsets.
        :param block: The block we wish to push.
        :param direction: A unit vector direction to push.
        :param timesteps: The number of timesteps to execute the action for.
        :param delta: How far to move each timestep.
        """
        block_pos = block.pose.pos
        self.start_pos = Position(x=block_pos.x - direction[0]*delta*20,
                                  y=block_pos.y - direction[1]*delta*20,
                                  z=block_pos.z - direction[2]*delta*20)
        self.c_id = p.createConstraint(parentBodyUniqueId=world.get_hand_id(),
                                       parentLinkIndex=-1,
                                       childBodyUniqueId=-1,
                                       childLinkIndex=-1,
                                       jointType=p.JOINT_FIXED,
                                       jointAxis=(0,0,0),
                                       parentFramePosition=(0, 0, 0),
                                       childFramePosition=self.start_pos)
        self.direction = direction
        self.timesteps = timesteps
        self.delta = delta
        self.world = world
        self.tx = 0
        
        # Store the world state when executing each action.
        self.trajectory = []

    def step(self):
        """ Move the hand forward by delta. """
        if self.tx < self.timesteps:
            updated_pos = Position(x=self.start_pos.x + self.tx*self.delta*self.direction[0],
                                   y=self.start_pos.y + self.tx*self.delta*self.direction[1],
                                   z=self.start_pos.z + self.tx*self.delta*self.direction[2])
            p.changeConstraint(userConstraintUniqueId=self.c_id, 
                               jointChildPivot=updated_pos)
            self.trajectory.append(self.world.get_positions())
            self.tx += 1
    
    @staticmethod
    def get_random_dir():
        angle = np.random.uniform(0, np.pi)
        return (np.cos(angle), np.sin(angle), 0)


def make_world(com):
    platform = Object(name='platform',
                      dimensions=Dimensions(x=0.3, y=0.2, z=0.05),
                      mass=100,
                      com=Position(x=0., y=0., z=0.),
                      color=Color(r=0.25, g=0.25, b=0.25))
    platform.set_pose(Pose(pos=Position(x=0., y=0., z=0.025),
                           orn=Quaternion(x=0, y=0, z=0, w=1)))

    block = Object(name='block',
                   dimensions=Dimensions(x=0.05, y=0.05, z=0.05),
                   mass=1,
                   com=com,
                   color=Color(r=1., g=0., b=0.))
    block.set_pose(Pose(pos=Position(x=0., y=0., z=0.075),
                        orn=Quaternion(x=0, y=0, z=0, w=1)))
  
    return World([platform, block])


if __name__ == '__main__':
    true_com = Position(x=0., y=0., z=0.)
    true_world = make_world(true_com)

    com_ranges = get_com_ranges(true_world.objects[1])
    com_particles, _ = create_uniform_particles(5, 3, com_ranges)
    particle_worlds = [make_world(particle) for particle in com_particles]

    env = Environment([true_world]+particle_worlds, vis_sim=True)
    
    d = PushAction.get_random_dir()

    actions = []
    for w in [true_world] + particle_worlds:
        action = PushAction(world=w, 
                            block=w.objects[1],
                            direction=d, 
                            timesteps=50)
        actions.append(action)

    for ix in range(100):
        env.step(actions=actions)
        
    print(actions[0].trajectory)


    
