from block_utils import Environment, World, Object, Position, Pose, \
                        Quaternion, Dimensions, Color, get_com_ranges, \
                        rotation_group, get_rotated_block
# from filter_utils import create_uniform_particles
import pybullet as p
import copy
import numpy as np
from operator import itemgetter


def plan_action(belief, k=3, exp_type='reduce_var'):
    """ Given a set of particles, choose the action that maximizes the observed variance.
    :param particle_block: A list of the current set of particles instantiated as blocks.
    :param k: Number of pushes to do for each orientation.
    """
    if exp_type == 'reduce_var':
        print('Finding variance reducing action')
        results = []
        for rot in rotation_group():
            for _ in range(k):
                # create a bunch of blocks with the same geometry where each COM
                # for each block is set to one of the particles
                particle_blocks = [copy.deepcopy(belief.block) for particle in belief.particles.particles]
                for (com, particle_block) in zip(belief.particles.particles, particle_blocks):
                    particle_block.com = com
                particle_worlds = [make_platform_world(pb, rot) for pb in particle_blocks]
                env = Environment(particle_worlds, vis_sim=False)
                action = PushAction(block_pos=particle_worlds[0].get_pose(particle_worlds[0].objects[1]).pos,
                                    direction=PushAction.get_random_dir(),
                                    timesteps=50)

                for t in range(50):
                    env.step(action=action)

                # Get end pose of all particle blocks.
                poses = np.array([w.get_pose(w.objects[1]).pos for w in particle_worlds])
                var = np.var(poses, axis=0)
                score = np.mean(var)
                print(var, score)
                results.append(((rot, action.direction), score))

                env.disconnect()
                env.cleanup()

        rot, direc = max(results, key=itemgetter(1))[0]
    else:
        print('Finding random action')
        rs = [r for r in rotation_group()]
        ix = np.random.choice(np.arange(len(rs)))
        rot = rs[ix]
        direc = PushAction.get_random_dir()

    return PushAction(direction=direc, rotation=rot)




class PushAction:
    def __init__(self, block_pos=None, direction=None, timesteps=50, rotation=None, delta=0.005):
        """ PushAction moves the hand in the given world by a fixed distance
            every timestep. We assume we will push the block in a direction through
            the object's geometric center.
        :param world: The world which this action should apply to. Used to get the hand
                      and calculate offsets.
        :param block_pos: The position of the block in a local world frame.
        :param direction: A unit vector direction to push.
        :param timesteps: The number of timesteps to execute the action for.
        :param delta: How far to move each timestep.
        """
        self.rotation = rotation
        self.direction = direction
        self.timesteps = timesteps
        self.delta = delta
        self.tx = 0
        if block_pos is not None:
            self.set_start_pos(block_pos)

    def step(self):
        """ Move the hand forward by delta.
        :return: The position of the hand in a local world frame.
        """
        t = self.tx
        if t > self.timesteps:
            t = self.timesteps
        updated_pos = Position(x=self.start_pos.x + t*self.delta*self.direction[0],
                               y=self.start_pos.y + t*self.delta*self.direction[1],
                               z=self.start_pos.z + t*self.delta*self.direction[2])
        self.tx += 1
        return updated_pos

    @staticmethod
    def get_random_dir():
        angle = np.random.uniform(0, np.pi)
        return (np.cos(angle), np.sin(angle), 0)

    def set_start_pos(self, block_pos):
        self.start_pos = Position(x=block_pos.x - self.direction[0]*self.delta*20,
                          y=block_pos.y - self.direction[1]*self.delta*20,
                          z=block_pos.z - self.direction[2]*self.delta*20)


def make_platform_world(p_block, rot):
    """ Given a block, create a world that has a platform to push that block off of.
    :param block: The Object which to place on the platform.
    """
    platform = Object(name='platform',
                      dimensions=Dimensions(x=0.3, y=0.2, z=0.05),
                      mass=100,
                      com=Position(x=0., y=0., z=0.),
                      color=Color(r=0.25, g=0.25, b=0.25))
    platform.set_pose(Pose(pos=Position(x=0., y=0., z=0.025),
                           orn=Quaternion(x=0, y=0, z=0, w=1)))

    p_block.set_pose(Pose(Position(0, 0, 0), Quaternion(*rot.as_quat())))

    block = get_rotated_block(p_block)
    block.set_pose(Pose(pos=Position(x=0., y=0., z=0.05+block.dimensions[2]/2.),
                        orn=Quaternion(0, 0, 0, 1)))

    return World([platform, block])


if __name__ == '__main__':
    true_com = Position(x=0., y=0., z=0.)

    block = Object(name='block',
                   dimensions=Dimensions(x=0.05, y=0.1, z=0.05),
                   mass=1,
                   com=true_com,
                   color=Color(r=1., g=0., b=0.))

    for r in rotation_group():
        true_world = make_platform_world(block, r)
        com_ranges = get_com_ranges(true_world.objects[1])
        com_particles, _ = create_uniform_particles(10, 3, com_ranges)

        particle_blocks = []
        for com in com_particles:
            particle_block = copy.deepcopy(block)
            particle_block.com = com
            particle_blocks.append(particle_block)

        particle_worlds = [make_platform_world(pb, r) for pb in particle_blocks]

        env = Environment([true_world]+particle_worlds, vis_sim=True)

        action = PushAction(block_pos=true_world.get_pose(true_world.objects[1]).pos,
                            direction=PushAction.get_random_dir(),
                            timesteps=50)

        for ix in range(100):
            env.step(action=action)
        p.disconnect()
