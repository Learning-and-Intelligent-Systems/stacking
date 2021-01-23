import pybullet as p
from time import sleep
import pybullet_data
from math import cos, pi, sin

physicsClient = p.connect(p.GUI)

p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)

p.setGravity(0, 0, -10)

planeOrn = [0,0,0,1]
planeId = p.loadURDF("geo/plane.urdf", [0,0,0], planeOrn)

#hollow= p.loadURDF("geo/cover_deform.urdf",[0,0,0.1], flags=p.URDF_USE_SELF_COLLISION)

sleep(3)
#tower demo
num_blocks=2
for i in range(num_blocks):
    # if i==num_blocks-1:
    #   innerbox=p.loadURDF("geo/block.urdf", [0.05,0.05,0.02+i*0.1], useMaximalCoordinates = True)
    # else:
    #   innerbox=p.loadURDF("geo/block.urdf", [0.05,0.05,0.02+i*0.1], useMaximalCoordinates = True)
    # hollow= p.loadURDF("geo/cover_deform.urdf",[0,0,0+i*0.1], flags=p.URDF_USE_SELF_COLLISION)
    hollow = p.loadSoftBody("geo/10cm_elm_3_solid_symmetric.vtk", basePosition = [0,0,0+i*0.1], mass = 0.1, useNeoHookean = 0, useBendingSprings=0, useMassSpring=1, springElasticStiffness=95, springDampingStiffness=.1, springDampingAllDirections = 1, useSelfCollision = 0, frictionCoeff = 1.1, useFaceContact=1) #50, 0.05 demo
    # if i==2:
    #   innerbox=p.loadURDF("geo/block.urdf", [0.5,0.5,0.5+i*1], useMaximalCoordinates = True)
    # else:
    #   innerbox=p.loadURDF("geo/block.urdf", [0.5,0.5,0.5+i*1], useMaximalCoordinates = True)


#cool experiment to show rigid v deformable com requirements
#shows how the com needs to change to maintain stability (no cover around -0.2 m but for deformables u need about -0.5 m)
# innerbox=p.loadURDF("geo/block.urdf", [0.5,0.5,0.5], useMaximalCoordinates = True)
# hollow= p.loadURDF("geo/cover_deform.urdf",[0,0,0], flags=p.URDF_USE_SELF_COLLISION)
# innerbox2=p.loadURDF("geo/block2.urdf", [1,0.5,1.5], useMaximalCoordinates = True)
# hollow2= p.loadURDF("geo/cover_deform.urdf", [0.5,0,1],flags=p.URDF_USE_SELF_COLLISION)

# #good example of tile on block, commented out clothId uses spring model
# boxId = p.loadURDF("geo/cube.urdf", [0,0,2.5], useMaximalCoordinates = True) #no rigid bodies
#clothId= p.loadURDF("geo/cover_deform.urdf", [-0.5,0.5,2], baseOrientation=[-0.5,-0.5,-0.5,0.5], flags=p.URDF_USE_SELF_COLLISION)
##clothId = p.loadSoftBody("geo/res_2_cover_6_in.vtk", basePosition = [-0.5,0.5,2], baseOrientation=[-0.5,-0.5,-0.5,0.5], mass = 1., useNeoHookean = 0, useBendingSprings=1, useMassSpring=1, springElasticStiffness=20, springDampingStiffness=.1, springDampingAllDirections = 1, useSelfCollision = 0, frictionCoeff = .5, useFaceContact=1)
# p.createSoftBodyAnchor(clothId ,0,boxId,4) #handpicked from meshAPI module and obj file
# p.createSoftBodyAnchor(clothId ,1,boxId,2)
# p.createSoftBodyAnchor(clothId ,2,boxId,3)
# p.createSoftBodyAnchor(clothId ,3,boxId,5)
# p.createSoftBodyAnchor(clothId ,10,boxId,8)
# p.createSoftBodyAnchor(clothId ,14,boxId,9)
# p.createSoftBodyAnchor(clothId ,18,boxId,10)
# p.createSoftBodyAnchor(clothId ,22,boxId,11)

#print(p.getMeshData(hollow, linkIndex=0))

#deformable demonstration
#bunny = p.loadURDF("geo/torus_deform.urdf", [0,0,1], flags=p.URDF_USE_SELF_COLLISION)
#bunny2 = p.loadURDF("geo/torus_deform.urdf", [0,0,3], flags=p.URDF_USE_SELF_COLLISION)


# #IN PROGRESS, will complete if rubber hollow block does not work
# def add_anchors(body1="geo/cover_deform.urdf", body2="geo/cube.urdf", **kwargs):
#   #anchors all 6 pads to block
#   count=0
#   base_pos, base_orientation=p.getBasePositionAndOrientation(body2)
#   #hard code quaternions list
#   #vertices for body2?
#   mesh_verts=[0,1,2,3,10,14,18,22] #face independent
#   while count<6:
#     x,y,z,w=quats[count] #face dependent
#     vertices=verts[count] #face dependent
#     #assumes body frames of both bodies are centered and length of body2 is 1m
#     cover= p.loadURDF(body1, basePosition=[base_pos[0],base_pos[1],base_pos[2]-0.5], baseOrientation=[x,y,z,w], flags=p.URDF_USE_SELF_COLLISION) #baseOrientation= base_orientation*[x,y,z,w] quat math
#     count2=0
#     while count<8:
#       p.createSoftBodyAnchor(cover,mesh_verts[count2],body2,vertices[count2])

p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25)
p.setRealTimeSimulation(1)

counter=0
while p.isConnected():
  p.stepSimulation()
  #p.setGravity(0, 0, -10)
  sleep(1./240.)


