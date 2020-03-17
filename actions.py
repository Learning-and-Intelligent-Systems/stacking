from block_utils import Environment, World, Object, Position, \
                        Dimensions, Color

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

    for _ in range(100):
        env.step()
        


    
