import pybullet as p #2.8.2
from odio_urdf import * #https://github.com/The-Ripp3r/odio_urdf
import os
import sys
import time


#constants: 
# -geo of all parts
# -cover is the same for both blocks
# -locations of blocks (for now block2 above block1 by 0.1 m)
# -collision margin
# -mass ratio of cover to inner is 1:1.6 for bottom block 
#   (comes from cover needing to be at least 1 and inner being 60% more in volume)

#ranges:
#m: [90, 570] step=5?
#l: [200, 570] step=5?
#d: [0.01, 0.09] step=0.005
#f: [0, 1] step=0.05
#mass: [1, 2.6] step=0.05 

#SETUP
physicsClient = p.connect(p.DIRECT)
p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
p.setGravity(0, 0, -10)
p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25)
planeOrn = [0,0,0,1]
planeId = p.loadURDF("geo/plane_base.urdf", [0,0,0], planeOrn)
num_blocks=2
num_steps=2000


#######CODE########
#filename for where data is consolidated, filename for cover, filename for upper inner block, mu, lambda, damping, friction, mass of second block
data_filename, filename1, filename2, m, l, d, f, mass = sys.argv[1:]
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
with open(filename1, 'w') as handle:
        handle.write(str(cover))
with open(filename2, 'w') as handle:
        handle.write(str(mut_block))


#load blocks
blocks=[]
for i in range(num_blocks):
    if i==num_blocks-1:
      innerbox=p.loadURDF(filename2, [0.5,0.5,0.5+i*1], useMaximalCoordinates = True)
    else:
      innerbox=p.loadURDF("geo/block.urdf", [0.5,0.5,0.5+i*1], useMaximalCoordinates = True)
    blocks.append(innerbox)
    hollow= p.loadURDF(filename1,[0,0,0+i*1], flags=p.URDF_USE_SELF_COLLISION)

#step simulation and record last known locations
last_loc=[]
for i in range(num_steps):
    p.stepSimulation()
    if i==num_steps-1:
      for block in blocks:
        last_loc.append(p.getBasePositionAndOrientation(block)[0][2])

#Save if stable
unstable=False
for loc in last_loc:
  if loc<0:
    print("UNSTABLE")
    unstable=True
    break
if not unstable:
  print("SUCCESS")
  #SAVE SOMEHOW
  with open(data_filename, 'a+') as handle:
        handle.write("{}".format((m,l,d,f,mass)))