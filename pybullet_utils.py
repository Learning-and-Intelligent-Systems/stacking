import numpy as np
import pybullet as p
import transformations as trans

class PyBulletServer():

    def __init__(self, vis, cameraDistance=0.3):
        if vis:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
        p.resetDebugVisualizerCamera(
            cameraDistance=cameraDistance,
            cameraYaw=45,
            cameraPitch=-15,
            cameraTargetPosition=(0., 0., 0.))
        p.setGravity(0, 0, -10)

    def step(self):
        p.stepSimulation(physicsClientId=self.client)

    def disconnect(self):
        p.disconnect(physicsClientId=self.client)

    def load_urdf(self, urdf_name, pos, orn=(0., 0., 0., 1.)):
        return p.loadURDF(urdf_name, pos, orn, physicsClientId=self.client)

    def get_pose(self, object_id):
        return p.getBasePositionAndOrientation(object_id, physicsClientId=self.client)

    def get_vel(self, object_id):
        return p.getBaseVelocity(object_id, physicsClientId=self.client)

    def vis_frame(self, pos, quat, length=0.2, lifeTime=0.4):
        """ This function visualizes a coordinate frame for the supplied frame where the
        red,green,blue lines correpsond to the x,y,z axes.
        :param p: a vector of length 3, position of the frame (x,y,z)
        :param q: a vector of length 4, quaternion of the frame (x,y,z,w)
        """
        new_x = transformation([length, 0.0, 0.0], pos, quat)
        new_y = transformation([0.0, length, 0.0], pos, quat)
        new_z = transformation([0.0, 0.0, length], pos, quat)

        p.addUserDebugLine(pos, new_x, [1,0,0], lifeTime=lifeTime, physicsClientId=self.client)
        p.addUserDebugLine(pos, new_y, [0,1,0], lifeTime=lifeTime, physicsClientId=self.client)
        p.addUserDebugLine(pos, new_z, [0,0,1], lifeTime=lifeTime, physicsClientId=self.client)

def pause():
    try:
        print('press any key to continue execution')
        while True:
            p.stepSimulation()
    except KeyboardInterrupt:
        print('trying to exit')
        return

### Geometric Helper Functions ###
def skew_symmetric(vec):
    """ Calculates the skew-symmetric matrix of the supplied 3 dimensional vector
    """
    i, j, k = vec
    return np.array([[0, -k, j], [k, 0, -i], [-j, i, 0]])

def adjoint_transformation(vel, translation_vec, quat, inverse=False):
    """ Converts a velocity from one frame into another
    :param vel: a vector of length 6, the velocity to be converted, [0:3] are the
            linear velocity terms and [3:6] are the angular velcoity terms
    :param translation_vec: vector of length 3, the translation (x,y,z) to the desired frame
    :param quat: vector of length 4, quaternion rotation to the desired frame
    :param inverse (optional): if True, inverts the translation_vec and quat
    """
    if inverse:
        translation_vec, quat = p.invertTransform(translation_vec, quat)

    R = p.getMatrixFromQuaternion(quat)
    R = np.reshape(R, (3,3))
    T = np.zeros((6,6))
    T[:3,:3] = R
    T[:3,3:] = skew_symmetric(translation_vec).dot(R)
    T[3:,3:] = R
    return np.dot(T, vel)

def transformation(pos, translation_vec, quat, inverse=False):
    """ Converts a position from one frame to another
    :param p: vector of length 3, position (x,y,z) in frame original frame
    :param translation_vec: vector of length 3, (x,y,z) from original frame to desired frame
    :param quat: vector of length 4, (x,y,z,w) rotation from original frame to desired frame
    :param inverse (optional): if True, inverts the translation_vec and quat
    """
    if inverse:
        translation_vec, quat = p.invertTransform(translation_vec, quat)
    R = p.getMatrixFromQuaternion(quat)
    R = np.reshape(R, (3,3))
    T = np.zeros((4,4))
    T[:3,:3] = R
    T[:3,3] = translation_vec
    T[3,3] = 1.0
    pos = np.concatenate([pos, [1]])
    new_pos = np.dot(T, pos)
    return new_pos[:3]

def quat_math(q0, q1, inv0=False, inv1=False):
    """ Performs addition and subtraction between quaternions
    :param q0: a vector of length 4, quaternion rotation (x,y,z,w)
    :param q1: a vector of length 4, quaternion rotation (x,y,z,w)
    :param inv0: if True, inverts q0
    :param inv1: if True, inverts q1

    Examples:
    to get the total rotation from going to q0 then q1: quat_math(q0,q1,False,False)
    to get the rotation from q1 to q0: quat_math(q0,q1,True,False)
    """
    if not isinstance(q0, np.ndarray):
        q0 = np.array(q0)
    if not isinstance(q1, np.ndarray):
        q1 = np.array(q1)
    q0 = to_transquat(q0)
    q0 = trans.unit_vector(q0)
    q1 = to_transquat(q1)
    q1 = trans.unit_vector(q1)
    if inv0:
        q0 = trans.quaternion_conjugate(q0)
    if inv1:
        q1 = trans.quaternion_conjugate(q1)
    res = trans.quaternion_multiply(q0,q1)
    return to_pyquat(res)

def to_transquat(pybullet_quat):
    """Convert quaternion from (x,y,z,w) returned from pybullet to
    (w,x,y,z) convention used by transformations.py"""
    return np.concatenate([[pybullet_quat[3]], pybullet_quat[:3]])

def to_pyquat(trans_quat):
    """Convert quaternion from (w,x,y,z) returned from transformations.py to
    (x,y,z,w) convention used by pybullet"""
    return np.concatenate([trans_quat[1:], [trans_quat[0]]])

def euler_from_quaternion(q):
    """Convert quaternion from (x,y,z,w) returned from pybullet to
    euler angles = (roll, pitch, yaw) convention used by transformations.py"""
    trans_quat = to_transquat(q)
    eul = trans.euler_from_quaternion(trans_quat)
    return eul

def random_quaternion(rand=None):
    trans_quat = trans.random_quaternion(rand)
    return to_pyquat(trans_quat)

def pose_to_matrix(point, q):
    """Convert a pose to a transformation matrix
    """
    EPS = np.finfo(float).eps * 4.0
    q = to_transquat(q)
    n = np.dot(q, q)
    if n < EPS:
        M = np.identity(4)
        M[:3, 3] = point
        return M
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    M = np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])
    M[:3, 3] = point
    return M

def quaternion_from_matrix(matrix, isprecise=False):
    trans_q = trans.quaternion_from_matrix(matrix)
    return to_pyquat(trans_q)

def quaternion_from_euler(roll, pitch, yaw):
    trans_q = trans.quaternion_from_euler(roll, pitch, yaw, 'rxyz')
    return to_pyquat(trans_q)
