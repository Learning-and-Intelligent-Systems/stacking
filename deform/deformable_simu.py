import pybullet as p
from time import sleep
import pybullet_data

physicsClient = p.connect(p.GUI)

#p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)

p.setGravity(0, 0, -10)

planeOrn = [0,0,0,1]
planeId = p.loadURDF("plane.urdf", [0,0,0], planeOrn)

#boxId = p.loadURDF("block.urdf", [0,0,2.5], useMaximalCoordinates = True) #no rigid bodies

#clothId = p.loadSoftBody("cloth_z_up.obj", basePosition = [0,0,3], scale = 0.5, mass = 1., useNeoHookean = 0, useBendingSprings=1,useMassSpring=1, springElasticStiffness=40, springDampingStiffness=.1, springDampingAllDirections = 1, useSelfCollision = 0, frictionCoeff = .5, useFaceContact=1)

#anchor demonstration
# p.createSoftBodyAnchor(clothId  ,0) #removed "extra" parameter
# p.createSoftBodyAnchor(clothId ,1,-1)
# p.createSoftBodyAnchor(clothId ,0,boxId,-1)# [0.5,-0.5,1])
# p.createSoftBodyAnchor(clothId ,1,boxId,-1)# [-0.5,-0.5,1])
# p.createSoftBodyAnchor(clothId ,2,boxId,-1)# [0.5,-0.5,1])
# p.createSoftBodyAnchor(clothId ,3,boxId,-1)# [-0.5,-0.5,1])


#boxId2 = p.loadURDF("cube.urdf", [0,0,4.5],useMaximalCoordinates = True) #no rigid bodies

#deformable demonstration
#bunny = p.loadURDF("block.urdf", [0,0,1], flags=p.URDF_USE_SELF_COLLISION)
bunny2 = p.loadURDF("torus_deform.urdf", [0,0,3], flags=p.URDF_USE_SELF_COLLISION)

#print(p.getMeshData(clothId, linkIndex=0))

p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25)
p.setRealTimeSimulation(1)

while p.isConnected():
  p.setGravity(0,0,-10)
  sleep(1./240.)