import pybullet as p 
from odio_urdf import *
import os

physicsClient = p.connect(p.GUI)
p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD) #this is for FEM based simulation

tmp_dir = 'tmp_urdfs'
if not os.path.isdir(tmp_dir):
    os.mkdir(tmp_dir)

def out_of_bounds(l, index):
    try:
        return l[index]
    except IndexError:
        return 0

def createDeformableHollowBlock(file="res_2_cover_8_in.vtk", **kwargs):
    #inertia takes a list with a specfic order but I can change it to a dicitonary if we desire
    deformable_urdf=Deformable(kwargs['name'])
    i=Inertial( Mass(value= kwargs['mass']),
                    Inertia(ixx=out_of_bounds(kwargs['inertia'], 0),
                            ixy=out_of_bounds(kwargs['inertia'], 1),
                            ixz=out_of_bounds(kwargs['inertia'], 2),
                            iyy=out_of_bounds(kwargs['inertia'], 3),
                            iyz=out_of_bounds(kwargs['inertia'], 4),
                            izz=out_of_bounds(kwargs['inertia'], 5))
                    )
    c=Collision_margin(value = kwargs['collision_margin']) #should be 0
    r=Repulsion_Stiffness(value = kwargs['repulsion_stiffness'])
    f=Friction(value = kwargs['friction']),
    n=Neohookean(mu=kwargs['mu'], lam = kwargs['lamda'], damping=kwargs['damping']),
    v=Visual(filename=file)

    return Robot(deformable_urdf(i,c,r,f,n,v))

if __name__ == "__main__":
    myRobot = Robot(Deformable(
            Inertial(
                Mass(value=1),
                Inertia(ixx=100, ixy=0),
            ),
            Collision_margin(value = 0.006),
            Repulsion_Stiffness(value = 800.0),
            Friction(value = 0.5),
            Neohookean(mu=200.0, lam = 200.0, damping=0.01),
            Visual2(filename="torus.vtk"),
            name="practice"
            ), name="block")

    print(myRobot)
    with open(tmp_dir+'/practice_'+ str(0) + '.urdf', 'w') as handle:
        handle.write(str(myRobot))