from distutils.util import execute
import IPython
import numpy as np
import os
import pb_robot
import panda_controls
from pybullet_object_models import ycb_objects
import pybullet as p
import time


if __name__ == '__main__':
    pb_robot.utils.connect(use_gui=True)
    pb_robot.utils.set_default_camera()

    floor_file = 'models/short_floor.urdf'
    floor = pb_robot.body.createBody(floor_file)

    mustard = pb_robot.body.createBody(os.path.join(ycb_objects.getDataPath(), 'YcbMustardBottle', 'model.urdf'))
    mustard.set_base_link_pose(((0, 0, pb_robot.placements.stable_z(mustard, floor)), (0, 0, 0, 1)))

    # robot = pb_robot.panda.Panda()
    # robot.hand.Open()
    # pc = panda_controls.PandaControls(robot.arm)
    hand = pb_robot.panda.PandaHand()
    init_pos = [0.01, -0.115, 0.1]
    init_orn = pb_robot.transformations.quaternion_from_euler(0, np.pi/2, np.pi/2)
    hand_control = pb_robot.panda_controls.FloatingHandControl(hand, init_pos, init_orn)
    hand_control.open()
    input()
    print(mustard.get_dynamics_info())
    mustard.set_dynamics(-1, lateralFriction=0.5, mass=1)
    # Disable velocity control motors.
    force = 10
    hand_control.close(force=force)

    for ix in range(0, 500):
        hand_control.move_to([init_pos[0], init_pos[1], init_pos[2] + 0.001*ix], init_orn, force)
        # p.setJointMotorControlArray(bodyUniqueId=hand.id,
        #                             jointIndices=[0, 1],
        #                             controlMode=p.TORQUE_CONTROL,
        #                             forces=[-force, -force])
        # p.changeConstraint(hand_control.cid, [init_pos[0], init_pos[1], init_pos[2] + 0.001*ix], jointChildFrameOrientation=init_orn, maxForce=500)
        # p.stepSimulation()
        # time.sleep(0.01)

    for _ in range(100):
        hand_control.move_to([init_pos[0], init_pos[1], init_pos[2] + 0.001*500], init_orn, force)

    IPython.embed()
    pb_robot.utils.wait_for_user()
    pb_robot.utils.disconnect()

