""" Massachusetts Institute of Technology

Izzybrand, 2020
"""
import numpy as np
from scipy.spatial.transform import Rotation


def eul_to_rot(theta):
    """ Conver a vector of euler angles to a rotation matrix
    
    Arguments:
        theta {np.ndarray} -- a 3-vector of euler angles (around x, y, z)
    
    Returns:
        np.ndarray -- 3x3 rotation matrix
    """

    return Rotation.from_euler('xyz', theta).as_matrix()

def rot_to_quat(R):
    return Rotation.from_matrix(R).as_quat()

def Rt_to_pose_matrix(R, t):
    """ convert a rotation and translation into a pose matrix
    
    Arguments:
        R {np.ndarray} -- 3x3 rotation matrix Nx4x4 array of rotation matrices
        t {np.ndarray} -- 3x1 translation vector Nx3 array of translations
    
    Returns:
        np.ndarray -- 4x4 pose matrix
    """
    if len(R.shape) == 2:
        P = np.eye(4)
        P[:3, :3] = R
        P[:3, 3] = t
        return P
    else:
        P = np.zeros([R.shape[0], 4, 4])
        P[:] = np.eye(4)
        P[:, :3, :3] = R
        P[:, :3, 3] = t
        return P

def pose_matrix_to_Rt(P):
    """ Take a 4x4 pose matrix and convert to a 3x3 rotation and 3x1 position
    
    Arguments:
        P {np.ndarray} -- 4x4 or Nx4x4 array of pose matrices
    
    Returns:
        (np.ndarray, np.ndarray) -- rotations and positions
    """
    if len(P.shape) == 2:
        return P[:3, :3], P[:3, 3]
    else:
        return P[:, :3, :3], P[:, :3, 3]

def snap_rotation_matrix(R):
    """ snap a 3x3 rotation matrix to the nearest 90 degree rotation
    
    
    Arguments:
        R {np.ndarray} -- 3x3 rotation matrix
    
    Returns:
        np.ndarray -- snapped rotation matrix
    """
    E = np.hstack([np.eye(3), -np.eye(3)])
    dot_with_axes = R @ E
    closest_axes = np.argmax(dot_with_axes, axis=1)
    r1 = E.T[closest_axes[0]]
    r2 = E.T[closest_axes[1]]
    r3 = np.cross(r1, r2)
    R_snapped = np.array([r1, r2, r3])
    return R_snapped

def mean_pose(poses):
    """ take the mean of 4x4 pose matricies

    The euclidean mean of positions and the chordal L2 mean of rotations
    
    Arguments:
        poses {np.ndarray} -- Nx4x4 pose matrices
    
    Returns:
        np.ndarray -- 4x4 pose matrix, the mean
    """
    Rs, Ts = pose_matrix_to_Rt(np.array(poses))
    R_mean = Rotation.from_matrix(Rs).mean().as_matrix()
    T_mean = Ts.mean(axis=0)
    return Rt_to_pose_matrix(R_mean, T_mean)
