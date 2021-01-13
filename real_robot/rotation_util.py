import numpy as np

def eul_to_rot(theta):
    R = [[np.cos(theta[1])*np.cos(theta[2]), np.sin(theta[0])*np.sin(theta[1])*np.cos(theta[2]) - np.sin(theta[2])*np.cos(theta[0]), np.sin(theta[1])*np.cos(theta[0])*np.cos(theta[2]) + np.sin(theta[0])*np.sin(theta[2])],
        [np.sin(theta[2])*np.cos(theta[1]), np.sin(theta[0])*np.sin(theta[1])*np.sin(theta[2]) + np.cos(theta[0])*np.cos(theta[2]), np.sin(theta[1])*np.sin(theta[2])*np.cos(theta[0]) - np.sin(theta[0])*np.cos(theta[2])],
        [-np.sin(theta[1]),                 np.sin(theta[0])*np.cos(theta[1]),                                                      np.cos(theta[0])*np.cos(theta[1])]]

    return R

def Rt_to_pose_matrix(R, t):
    """ convert a rotation and translation into a pose matrix
    
    Arguments:
        R {np.ndarray} -- 3x3 rotation matrix
        t {np.ndarray} -- 3x1 translation vector
    
    Returns:
        np.ndarray -- 4x4 pose matrix
    """
    P = np.eye(4)
    P[:3, :3] = R
    P[:3, 3] = t
    return P

def pose_matrix_to_Rt(P):
    return P[:3, :3], P[:3, 3]
