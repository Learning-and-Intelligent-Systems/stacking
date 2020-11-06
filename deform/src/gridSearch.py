import pybullet as p
from odio_urdf import *
import os
import pybullet_data

physicsClient = p.connect(p.GUI)
p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
p.setGravity(0, 0, -10)
p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25)

planeOrn = [0,0,0,1]
planeId = p.loadURDF("geo/plane_base.urdf", [0,0,0], planeOrn)

num_blocks=2
num_steps=2000
approved_combos=[] # tuples of (mu, lambda, damping, friction, and inner mass of second block)

#constants: 
# -geo of all parts
# -cover is the same for both blocks
# -locations of blocks (for now block2 above block1 by 0.1 m)
# -collision margin
# -mass ratio of cover to inner is 1:1.6 for bottom block 
#   (comes from cover needing to be at least 1 and inner being 60% more in volume)

#for loops for mu, lambda, damping, friction, inner mass of second block [1,2]
m=570
l=570
d=0.01
f=0.5
mass=1.6

#generate urdfs
cover = Robot(Deformable(
            Inertial(
                Mass(value=1),
                Inertia(ixx=0, ixy=0, ixz=0, iyy=0, iyz=0, izz=0),
            ),
            Collision_margin(value = 0.01),
            Repulsion_Stiffness(value = 800.0),
            Friction(value = f),
            Neohookean(mu=m, lam = l, damping=d),
            Visual2(filename="res_2_cover_6_in.vtk"),
            name="p1"
            ), name="cover")

mut_block = Robot(Link(
            Inertial(
                Origin(xyz= (0,0,0), rpy= (0,0,0)),
                Mass(value=mass),
                Inertia(ixx=0.001, ixy=0, ixz=0, iyy=0.001, iyz=0, izz=0.001),
            ),
            Collision(
              Origin(xyz= (0,0,0), rpy= (0,0,0)),
              Geometry(
                Box(size=(0.6,0.6,0.6)),
              ),
            ),
            Visual(
              Origin(xyz= (0,0,0), rpy= (0,0,0)),
              Geometry(
                Box(size=(0.6,0.6,0.6)),
              ),
              Material(
                "color",
                Color(rgba=(0,1,1,1)),
              ),
            ),
            name="p2"
            ), name="mut_block")

#save urdfs
with open('geo/mutable_flex.urdf', 'w') as handle:
        handle.write(str(cover))
with open('geo/mutable_block.urdf', 'w') as handle:
        handle.write(str(mut_block))


#load blocks
blocks=[]
for i in range(num_blocks):
    if i==num_blocks-1:
      innerbox=p.loadURDF("geo/mutable_block.urdf", [0.5,0.5,0.5+i*1], useMaximalCoordinates = True)
    else:
      innerbox=p.loadURDF("geo/block.urdf", [0.5,0.5,0.5+i*1], useMaximalCoordinates = True)
    blocks.append(innerbox)
    hollow= p.loadURDF("geo/mutable_flex.urdf",[0,0,0+i*1], flags=p.URDF_USE_SELF_COLLISION)

#step simulation and record last known locations
last_loc=[]
for i in range(num_steps):
    p.stepSimulation()
    if i==num_steps-1:
      for block in blocks:
        last_loc.append(p.getBasePositionAndOrientation(block)[0][2])

unstable=False
for loc in last_loc:
  if loc<0:
    print("UNSTABLE")
    unstable=True
    break
if not unstable:
  print("SUCCESS")
  approved_combos.append((m,l,d,f,mass))