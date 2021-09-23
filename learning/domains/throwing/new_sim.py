import numpy as np
import pybullet as p
import pybullet_data
import time


class ThrowingSimulator:
    def __init__(self, objects, vis=False, dt=0.0005, tmax=5):
        self.tmax = tmax
        self.objects = objects
        self.vis = vis
        self.obstacles = np.array([[0.06, 1, 0.06, 0.9, 0, -0.01, 0., np.pi/4., 0.]]) # dimensions || position || roll-pitch-yaw

        self.stop_vel_thresh = 0.01      # Maximum linear velocity before stopping simulation [m/s]
        self.stop_vel_count = 10         # Number of consecutive counts below velocity threshold before simulation is stopped

        self.setup()
        self.trace = False

    def setup(self):
        # set up the simulator
        client = p.connect(p.GUI if self.vis else p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) # turn off debug windows
        p.setGravity(0, 0, -9.81, physicsClientId=client) # set gravity
        # simulation tries to run at real time only if visualizer is enabled
        p.setRealTimeSimulation(False)

        p.resetDebugVisualizerCamera(cameraDistance=2.,
                         cameraYaw=-60,
                         cameraPitch=-30,
                         cameraTargetPosition=[1, 0, 0])

        # load models
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        planeId = p.loadURDF("plane.urdf") # load the groud plane
        p.changeDynamics(planeId, -1, restitution=1)

        # and create obstacles
        for obs in self.obstacles:
            obsId = p.createMultiBody(0,
                p.createCollisionShape(p.GEOM_BOX, halfExtents=obs[0:3]/2),
                p.createVisualShape(p.GEOM_BOX, halfExtents=obs[0:3]/2, rgbaColor=np.ones(4)),
                basePosition=obs[3:6],
                baseOrientation=p.getQuaternionFromEuler(obs[6:9]))
            p.changeDynamics(obsId, -1, restitution=1)

    def disconnect(self):
        p.disconnect()

    def get_state(self, p, ballId):
        (x, y, z), orientation = p.getBasePositionAndOrientation(ballId)
        rx, ry, rz = p.getEulerFromQuaternion(orientation)
        (vx, vy, vz), (wx, wy, wz) = p.getBaseVelocity(ballId)
        state = np.array([x, z, ry, vx, vz, wy])
        return state

    def simulate(self, action):
        b = action.object

        # create a sphere
        ballId = p.createMultiBody(b.mass,
            p.createCollisionShape(p.GEOM_SPHERE, radius=b.radius),
            p.createVisualShape(p.GEOM_SPHERE, radius=b.radius, rgbaColor=list(b.color) + [1]))

        # set the dynamics parameters
        p.changeDynamics(ballId, -1, restitution=b.bounciness,
                                     rollingFriction=b.rolling_resistance,
                                     lateralFriction=b.friction_coef,
                                     linearDamping=b.air_drag_linear,
                                     angularDamping=b.air_drag_angular)

        # set the throw initial position
        p.resetBasePositionAndOrientation(ballId, [action.x, 0, action.y], [0,0,0,1])
        p.resetBaseVelocity(ballId, [action.vx, 0, action.vy], [0, action.w, 0])


        dt = 1/240.
        results = {
            "time": [],
            "state": [],
            "num_bounces": 0
        }

        stop_count = 0
        trace_ids = []

        for t in np.arange(0, self.tmax, dt):
            p.stepSimulation()
            results["state"].append(self.get_state(p, ballId))
            results["time"].append(t)

            if self.trace:
                trace_ids.append(p.createMultiBody(0, baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.01, rgbaColor=[1,0,0,0.2]),
                    basePosition=p.getBasePositionAndOrientation(ballId)[0]))

            v = np.linalg.norm(results["state"][-1][3:5])
            if v < self.stop_vel_thresh:
                stop_count += 1
                if stop_count > self.stop_vel_count:
                    break
            else:
                stop_count = 0


            if self.vis: time.sleep(dt)

        p.removeBody(ballId)
        for t_id in trace_ids:
            p.removeBody(t)

        results["time"] = np.array(results["time"]).T
        results["state"] = np.array(results["state"]).T
        return results
