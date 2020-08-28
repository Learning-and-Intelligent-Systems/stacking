import pybullet as p
from time import sleep
import pybullet_data
from math import cos, pi, sin

physicsClient = p.connect(p.GUI)

#p.setAdditionalSearchPath(pybullet_data.getD ataPath())

p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)

p.setGravity(0, 0, -10)

planeOrn = [0,0,0,1]
planeId = p.loadURDF("geo/plane.urdf", [0,0,0], planeOrn)

#good example of tile on block, commented out clothId uses spring model
boxId = p.loadURDF("geo/cube.urdf", [0,0,2.5], useMaximalCoordinates = True) #no rigid bodies
clothId= p.loadURDF("geo/cover_deform.urdf", [-0.5,0.5,2], baseOrientation=[-0.5,-0.5,-0.5,0.5], flags=p.URDF_USE_SELF_COLLISION)
#clothId = p.loadSoftBody("geo/cover.vtk", basePosition = [-0.5,0.5,2], baseOrientation=[-0.5,-0.5,-0.5,0.5], mass = 1., useNeoHookean = 0, useBendingSprings=1, useMassSpring=1, springElasticStiffness=20, springDampingStiffness=.1, springDampingAllDirections = 1, useSelfCollision = 0, frictionCoeff = .5, useFaceContact=1)
p.createSoftBodyAnchor(clothId ,0,boxId,4) #handpicked from meshAPI module and obj file
p.createSoftBodyAnchor(clothId ,1,boxId,2)
p.createSoftBodyAnchor(clothId ,2,boxId,3)
p.createSoftBodyAnchor(clothId ,3,boxId,5)
p.createSoftBodyAnchor(clothId ,10,boxId,8)
p.createSoftBodyAnchor(clothId ,14,boxId,9)
p.createSoftBodyAnchor(clothId ,18,boxId,10)
p.createSoftBodyAnchor(clothId ,22,boxId,11)

#print(p.getMeshData(clothId, linkIndex=0))

#deformable demonstration
# bunny = p.loadURDF("geo/block.urdf", [0,0,1], flags=p.URDF_USE_SELF_COLLISION)
# bunny2 = p.loadURDF("geo/torus_deform.urdf", [0,0,3], flags=p.URDF_USE_SELF_COLLISION)


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

while p.isConnected():
  p.setGravity(0,0,-10)
  sleep(1./240.)