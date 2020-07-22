import argparse
import copy
import numpy as np
from operator import itemgetter
import pybullet as p

from base_class import ActionBase
from block_utils import Environment, World, Object, Position, Pose, \
                        Quaternion, Dimensions, Color, \
                        rotation_group, get_com_ranges, \
                        ZERO_POS, ZERO_ROT
from filter_utils import create_uniform_particles, make_platform_world

def plan_action(belief, k=1, exp_type='reduce_var', action_type='push'):
    """ Given a set of particles, choose the action that maximizes the observed variance.
    :param particle_block: A list of the current set of particles instantiated as blocks.
    :param k: Number of pushes to do for each orientation.
    """
    if action_type == 'push':
        if exp_type == 'reduce_var':
            print('Finding variance reducing push action')
            results = []
            for rot in rotation_group():
                for _ in range(k):
                    action = PushAction(rot=rot, timesteps=50, block=belief.block)

                    # create a bunch of blocks with the same geometry where each COM
                    # for each block is set to one of the particles
                    particle_blocks = [copy.deepcopy(belief.block) for particle in belief.particles.particles]
                    for (com, particle_block) in zip(belief.particles.particles, particle_blocks):
                        particle_block.com = com
                    particle_worlds = [make_platform_world(pb, action) for pb in particle_blocks]
                    env = Environment(particle_worlds, vis_sim=False)

                    # get the the position of the block to set the start position of the push action
                    # action.set_push_start_pos(particle_worlds[0].get_pose(particle_worlds[0].objects[1]).pos)

                    for _ in range(50):
                        env.step(action=action)

                    # Get end pose of all particle blocks.
                    poses = np.array([w.get_pose(w.objects[1]).pos for w in particle_worlds])
                    var = np.var(poses, axis=0)
                    score = np.mean(var)
                    print(var, score)
                    results.append((action, score))

                    env.disconnect()
                    env.cleanup()

            return max(results, key=itemgetter(1))[0]

        else: 
            print('Finding random push action')
            rs = [r for r in rotation_group()]
            ix = np.random.choice(np.arange(len(rs)))
            rot = rs[ix]
            push_action = PushAction(rot=rot, block=belief.block)

        return push_action

    else:
        if exp_type == 'reduce_var':
            print('Finding variance reducing place action')
            results = []
            for rot in rotation_group():
                for _ in range(k):
                    action = PlaceAction(rot=rot, pos=None, block=belief.block)

                    # create a bunch of blocks with the same geometry where each COM
                    # for each block is set to one of the particles
                    particle_blocks = [copy.deepcopy(belief.block) for particle in belief.particles.particles]
                    for (com, particle_block) in zip(belief.particles.particles, particle_blocks):
                        particle_block.com = com
                    particle_worlds = [make_platform_world(pb, action) for pb in particle_blocks]
                    env = Environment(particle_worlds, vis_sim=False)

                    # get the the position of the block to set the start position of the push action
                    # action.set_push_start_pos(particle_worlds[0].get_pose(particle_worlds[0].objects[1]).pos)

                    for _ in range(50):
                        env.step(action=action)
                    # Get end pose of all particle blocks.
                    poses = np.array([w.get_pose(w.objects[1]).pos for w in particle_worlds])
                    var = np.var(poses, axis=0)
                    score = np.max(var)
                    cov = np.cov(poses, rowvar=False)
                    w, v = np.linalg.eig(cov)
                    #print(w)
                    score = np.max(w)
                    #print(var, score)
                    results.append((action, score))

                    env.disconnect()
                    env.cleanup()

            return max(results, key=itemgetter(1))[0]
        else:
            print('Finding random place action')
            # pick a random rotation
            rs = [r for r in rotation_group()]
            ix = np.random.choice(np.arange(len(rs)))
            rot = rs[ix]
            
            # construct the corresponding place action
            place_action = PlaceAction(rot=rot, pos=None, block=belief.block)
            return place_action


class PushAction(ActionBase):
    def __init__(self, direction=None, timesteps=50, rot=None, delta=0.005, block=None):
        """ PushAction moves the hand in the given world by a fixed distance
            every timestep. We assume we will push the block in a direction through
            the object's geometric center.
        :param world: The world which this action should apply to. Used to get the hand
                      and calculate offsets.
        :param direction: A unit vector direction to push.
        :param timesteps: The number of timesteps to execute the action for.
        :param delta: How far to move each timestep.
        """
        super(PushAction, self).__init__(T=timesteps)

        self.rot = rot
        self.timesteps = timesteps
        self.delta = delta
        self.tx = 0

        if direction is None:
            self.direction = self.get_random_dir()
        else:
            self.direction = direction

        _, _, block_height = np.abs(rot.apply(block.dimensions))

        table, leg = Object.platform()
        platform_height = table.dimensions.z + leg.dimensions.z

        self.push_start_pos = Position(x=0 - self.direction[0]*self.delta*20,
                                       y=0 - self.direction[1]*self.delta*20,
                                       z=platform_height + block_height/2 \
                                           - self.direction[2]*self.delta*20)


    def step(self):
        """ Move the hand forward by delta.
        :return: The position of the hand in a local world frame.
        """
        t = self.tx
        if t > self.timesteps:
            t = self.timesteps
        push_pos = Position(x=self.push_start_pos.x + t*self.delta*self.direction[0],
                            y=self.push_start_pos.y + t*self.delta*self.direction[1],
                            z=self.push_start_pos.z + t*self.delta*self.direction[2])
        self.tx += 1
        return push_pos

    def get_random_dir(self):
        angle = np.random.uniform(0, np.pi)
        return (np.cos(angle), np.sin(angle), 0)


class PlaceAction(ActionBase):
    def __init__(self, pos=None, rot=None, block=None):
        """ place_action simply specifies the desired intial block position
        """
        super(PlaceAction, self).__init__()
        if pos is None:
            pos = self.get_random_pos(rot, block)

        self.pos = pos
        self.rot = rot
        

    def get_random_pos(self, rot, block):
        """ Sample a random offset from a single corner of the platform.
        :param rot: The rotation the block will be placed with.
        :param block: Used to get the dimensions of the block to place.
        """
        # rotate the block by that rotation to get the dimensions in x,y
        new_dims = np.abs(rot.apply(block.dimensions))
        # and sample a position within the block to place on the corner of the platform
        place_pos = new_dims*(np.random.rand(3) - 0.5)
        table, _ = Object.platform()
        x, y, _ = place_pos + np.array(table.dimensions)/2
        return Position(x, y, 0)


"""
Test the specified action by creating a test world and performing a random action.
We also test here the the same action is applied across all particle worlds.
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action-type', choices=['push', 'place'], required=True)
    parser.add_argument('--agent-type', choices=['teleport', 'panda'], required=True)
    args = parser.parse_args()

    # Create the block.
    true_com = Position(x=0., y=0., z=0.)
    block = Object(name='block',
                   dimensions=Dimensions(x=0.05, y=0.1, z=0.05),
                   mass=1,
                   com=true_com,
                   color=Color(r=1., g=0., b=0.))

    # Test the action for all rotations.
    for r in rotation_group():
        # Create the action.
        if args.action_type == 'push':
            action = PushAction(direction=None, 
                                timesteps=50, 
                                rot=r, 
                                block=block)
        elif args.action_type == 'place':
            action = PlaceAction(pos=None,
                                 rot=r)
        else:
            raise NotImplementedError()

        # Make worlds for each block and test using different CoMs.
        true_world = make_platform_world(block, action)
        com_ranges = get_com_ranges(true_world.objects[1])
        com_particles, _ = create_uniform_particles(10, 3, com_ranges)

        particle_blocks = []
        for com in com_particles:
            particle_block = copy.deepcopy(block)
            particle_block.com = com
            particle_blocks.append(particle_block)

        particle_worlds = [make_platform_world(pb, action) for pb in particle_blocks]

        env = Environment([true_world]+particle_worlds, vis_sim=True)

        for ix in range(100):
            env.step(action=action)
        p.disconnect()
