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

#boxId = p.loadURDF("geo/block.urdf", [0,0,2.5], useMaximalCoordinates = True) #no rigid bodies

#difference between spring and neohookean
clothId= p.loadURDF("geo/torus_deform.urdf", [0,0,0], baseOrientation=[sin(pi/4),0,0,cos(pi/4)], flags=p.URDF_USE_SELF_COLLISION)
#clothId = p.loadSoftBody("cover.vtk", basePosition = [-0.5,-0.5,2], mass = 1., useNeoHookean = 0, useBendingSprings=1,useMassSpring=1, springElasticStiffness=20, springDampingStiffness=.1, springDampingAllDirections = 1, useSelfCollision = 0, frictionCoeff = .5, useFaceContact=1)
# p.createSoftBodyAnchor(clothId ,0,boxId,-1)# [0.5,-0.5,1])
# p.createSoftBodyAnchor(clothId ,1,boxId,-1)# [-0.5,-0.5,1])
# p.createSoftBodyAnchor(clothId ,2,boxId,-1)# [0.5,-0.5,1])
# p.createSoftBodyAnchor(clothId ,3,boxId,-1)# [-0.5,-0.5,1])
# p.createSoftBodyAnchor(clothId ,10,boxId,-1)# [-0.5,-0.5,1])
# p.createSoftBodyAnchor(clothId ,14,boxId,-1)# [-0.5,-0.5,1])
# p.createSoftBodyAnchor(clothId ,18,boxId,-1)# [-0.5,-0.5,1])
# p.createSoftBodyAnchor(clothId ,22,boxId,-1)# [-0.5,-0.5,1])

#clothId= p.loadURDF("geo/torus_deform.urdf", [-0.5,-0.5,2], baseOrientation=[1,0,0,1], flags=p.URDF_USE_SELF_COLLISION)
#clothId = p.loadSoftBody("cover.vtk", basePosition = [-0.5,-0.5,2], baseOrientation=[1,0,0,cos(0)], mass = 1., useNeoHookean = 0, useBendingSprings=1, useMassSpring=1, springElasticStiffness=20, springDampingStiffness=.1, springDampingAllDirections = 1, useSelfCollision = 0, frictionCoeff = .5, useFaceContact=1)
# p.createSoftBodyAnchor(clothId ,0,boxId,-1)# [0.5,-0.5,1])
# p.createSoftBodyAnchor(clothId ,1,boxId,-1)# [-0.5,-0.5,1])
# p.createSoftBodyAnchor(clothId ,2,boxId,-1)# [0.5,-0.5,1])
# p.createSoftBodyAnchor(clothId ,3,boxId,-1)# [-0.5,-0.5,1])
# p.createSoftBodyAnchor(clothId ,10,boxId,-1)# [-0.5,-0.5,1])
# p.createSoftBodyAnchor(clothId ,14,boxId,-1)# [-0.5,-0.5,1])
# p.createSoftBodyAnchor(clothId ,18,boxId,-1)# [-0.5,-0.5,1])
# p.createSoftBodyAnchor(clothId ,22,boxId,-1)# [-0.5,-0.5,1])

# clothId= p.loadURDF("geo/torus_deform.urdf", [-0.5,-0.5,3], baseOrientation=[0,1,0,cos(0)], flags=p.URDF_USE_SELF_COLLISION)
# #clothId = p.loadSoftBody("cover.vtk", basePosition = [-0.5,-0.5,2], baseOrientation=[1,0,0,cos(0)], mass = 1., useNeoHookean = 0, useBendingSprings=1, useMassSpring=1, springElasticStiffness=20, springDampingStiffness=.1, springDampingAllDirections = 1, useSelfCollision = 0, frictionCoeff = .5, useFaceContact=1)
# p.createSoftBodyAnchor(clothId ,0,boxId,-1)# [0.5,-0.5,1])
# p.createSoftBodyAnchor(clothId ,1,boxId,-1)# [-0.5,-0.5,1])
# p.createSoftBodyAnchor(clothId ,2,boxId,-1)# [0.5,-0.5,1])
# p.createSoftBodyAnchor(clothId ,3,boxId,-1)# [-0.5,-0.5,1])
# p.createSoftBodyAnchor(clothId ,10,boxId,-1)# [-0.5,-0.5,1])
# p.createSoftBodyAnchor(clothId ,14,boxId,-1)# [-0.5,-0.5,1])
# p.createSoftBodyAnchor(clothId ,18,boxId,-1)# [-0.5,-0.5,1])
# p.createSoftBodyAnchor(clothId ,22,boxId,-1)# [-0.5,-0.5,1])


def add_anchors(body, **kwargs):
  #anchors all 6 pads to block
  count=0
  base_pos, base_orientation=p.getBasePositionAndOrientation(boxId)
  offsets=kwargs['position_offsets']
  while count<6:
    dx,dy,dz=offsets.pop()
    if dy>0:
      pass
    #replace clothId with generate urdf that takes in position, orientation, and nehookean param
    clothId = p.loadSoftBody("cover.vtk", basePosition = [base_pos[0]+dx,base_pos[1]+dy,base_pos[2]+dz], mass = 1., useNeoHookean = 0, useBendingSprings=1,useMassSpring=1, springElasticStiffness=40, springDampingStiffness=.1, springDampingAllDirections = 1, useSelfCollision = 0, frictionCoeff = .5, useFaceContact=1)
    p.createSoftBodyAnchor(clothId ,0,boxId,-1)# [0.5,-0.5,1])
    p.createSoftBodyAnchor(clothId ,1,boxId,-1)# [-0.5,-0.5,1])
    p.createSoftBodyAnchor(clothId ,2,boxId,-1)# [0.5,-0.5,1])
    p.createSoftBodyAnchor(clothId ,3,boxId,-1)# [-0.5,-0.5,1])

#anchor demonstration
# p.createSoftBodyAnchor(clothId  ,0) #removed "extra" parameter
# p.createSoftBodyAnchor(clothId ,1,-1)
#boxId2 = p.loadURDF("cube.urdf", [0,0,4.5],useMaximalCoordinates = True) #no rigid bodies

#deformable demonstration
#bunny = p.loadURDF("block.urdf", [0,0,1], flags=p.URDF_USE_SELF_COLLISION)
#bunny2 = p.loadURDF("torus_deform.urdf", [0,0,3], flags=p.URDF_USE_SELF_COLLISION)

# print(p.getMeshData(clothId, linkIndex=0))

p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25)
p.setRealTimeSimulation(1)

while p.isConnected():
  p.setGravity(0,0,-10)
  sleep(1./240.)