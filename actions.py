from block_utils import Environment, World, Object, Position, \
                        Dimensions, Color
import pybullet as p
import copy


class PushAction:
    def __init__(self, world, direction, start_pos, timesteps, delta=0.005):
        self.start_pos = Position(x=world.offset[0]+start_pos.x,
                                  y=world.offset[1]+start_pos.y,
                                  z=start_pos.z)
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
        self.tx =0

    def step(self):
        if self.tx < self.timesteps:
            updated_pos = Position(x=self.start_pos.x + self.tx*self.delta*self.direction[0],
                                   y=self.start_pos.y + self.tx*self.delta*self.direction[1],
                                   z=self.start_pos.z + self.tx*self.delta*self.direction[2])
            p.changeConstraint(userConstraintUniqueId=self.c_id, 
                               jointChildPivot=updated_pos)
            self.tx += 1


if __name__ == '__main__':
    platform = Object(name='platform',
                      dimensions=Dimensions(x=0.3, y=0.2, z=0.05),
                      mass=100,
                      com=Position(x=0., y=0., z=0.),
                      color=Color(r=0.25, g=0.25, b=0.25))
    platform.set_pose(Position(x=0., y=0., z=0.025))

    block = Object(name='block',
                   dimensions=Dimensions(x=0.05, y=0.05, z=0.05),
                   mass=1,
                   com=Position(x=0., y=0., z=0.),
                   color=Color(r=1., g=0., b=0.))
    block.set_pose(Position(x=0., y=0., z=0.075))
  
    world = World([platform, block])
    env = Environment([world], vis_sim=True)
    
    action = PushAction(world=world, 
                        direction=(1, 0.1, 0), 
                        start_pos=Position(-0.1, 0, 0.075),
                        timesteps=50)

    for ix in range(100):
        env.step(actions=[action])
        


    
