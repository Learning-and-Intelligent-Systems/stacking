import pybullet as p 
from odio_urdf import *

physicsClient = p.connect(p.GUI)
p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD) #this is for FEM based simulation



def createBlock(name="dBlock1", origin=(0,0,0,0,0,0), mass=1.0, inertia=(0.001,0,0,0.001,0,0.001), collision_margin = 0.006,
 repulsion_stiffness=800.0, friction=0.5, neo=(60, 200, 0.01), size=(0.04,0.04,0.04), color=(0,1,1,1)):
    robot=Robot("block")
    deformable=Link(name)
    origin=Origin(xyz="{} {} {}".format(origin[0], origin[1], origin[2]), rpy="{} {} {}".format(origin[3], origin[4], origin[5]))
    inert=Inertial(origin,
                    Mass(value= "{}".format(mass)),
                    Inertia(ixx="{}".format(inertia[0]),
                            ixy="{}".format(inertia[1]),
                            ixz="{}".format(inertia[2]),
                            iyy="{}".format(inertia[3]),
                            iyz="{}".format(inertia[4]),
                            izz="{}".format(inertia[5]))
                    )
    collision = Collision(origin,
                        Geometry(Box(size="{} {} {}".format(size[0],size[1], size[2])))
                        )
    visual=Visual(origin,
                Geometry(Box(size="{} {} {}".format(size[0],size[1], size[2]))),
                Material(Color(rgba="{} {} {} {}".format(color[0], color[1], color[2], color[3])))
    )   
    
    print(robot(deformable(inert, collision, visual)))

#other hacky method

def createSpring():
    #this might be just loading a premade urdf file that is a spring
    pass

def createSoftBody():
    pass
