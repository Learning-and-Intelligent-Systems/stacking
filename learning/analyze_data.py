import numpy as np

#rom learning.train_gat import load_dataset

def is_com_stable(block0, block1):
    """
    Is the center of block1 on top of the support of block0?
    :param block0: Tensor representation of block0.
    :param block1: Tensor representation of block1.
    (m, com_x, com_y, com_z, dim_x, dim_y, dim_z, pos_x, pos_y, pos_z,...)
    """
    b1_cx, b1_cy = block1[1:3]
    b1_px, b1_py = block1[7:9]
    b0_dx, b0_dy = block0[4:6]

    if (np.abs(b1_px+b1_cx) > b0_dx/2) or (np.abs(b1_py+b1_cy) > b0_dy/2):
        return False
    return True

def get_geometric_thresholds(block0, block1):
    """
    Is the center of block1 on top of the support of block0?
    :param block0: Tensor representation of block0.
    :param block1: Tensor representation of block1.
    (m, com_x, com_y, com_z, dim_x, dim_y, dim_z, pos_x, pos_y, pos_z,...)
    """
    b1_px, b1_py = block1[7:9]
    b0_dx, b0_dy = block0[4:6]

    return np.abs(b1_px) - b0_dx/2, np.abs(b1_py) - b0_dy/2
        
def is_geometrically_stable(block0, block1):
    """
    Is the center of block1 on top of the support of block0?
    :param block0: Tensor representation of block0.
    :param block1: Tensor representation of block1.
    (m, com_x, com_y, com_z, dim_x, dim_y, dim_z, pos_x, pos_y, pos_z,...)
    """
    b1_px, b1_py = block1[7:9]
    b0_dx, b0_dy = block0[4:6]

    if (np.abs(b1_px) > b0_dx/2) or (np.abs(b1_py) > b0_dy/2):
        return False
    return True

def check_pairwise_stable(tower):
    """
    Return true if all the pairwise relations between the blocks are
    stable.
    """
    pass

def check_stability_type(dataset):
    """
    Check how many of the two-block towers are geometrically 
    stable or unstable. And check if this agrees with the true 
    stability label.
    """
    n_gstable, n_cstable = 0, 0

    n_gcstable = 0

    label_com = 0

    tensors, labels = dataset[0].tensors
    for ix in range(tensors.shape[0]):
        g_stable = is_geometrically_stable(tensors[ix, 0, :],
                                           tensors[ix, 1, :])
        c_stable = is_com_stable(tensors[ix, 0, :],
                                 tensors[ix, 1, :])
        #labels[ix]
        if c_stable and g_stable:
            n_gcstable += 1
        
        if c_stable == labels[ix]:
            label_com += 1
        
        n_gstable += g_stable
        n_cstable += c_stable
    
    print('Geometrically Stable:', n_gstable)
    print('CoM Stable:', n_cstable)
    print('Both:', n_gcstable)
    print('Valid:', label_com)
    

if __name__ == '__main__':
    FNAME = 'random_blocks_(x20000)_quat.pkl'
    dataset = load_dataset(FNAME)

    check_stability_type(dataset)
