import numpy as np
import pickle
import matplotlib.pyplot as plt
from block_utils import Object, Pose, Position, Quaternion
from tower_planner import TowerPlanner
#rom learning.train_graph_net import load_dataset

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
    
def to_blocks(tower):
    blocks = []
    for ix in range(tower.shape[0]):
        block = Object('block',
                       dimensions=tower[ix, 4:7].numpy().tolist(),
                       mass=tower[ix,0].item(),
                       com=tower[ix, 1:4].numpy().tolist(),
                       color=(1,0,0))
        block.pose = Pose(Position(*tower[ix, 7:10].numpy().tolist()),
                          Quaternion(*tower[ix, 10:14].numpy().tolist()))
        blocks.append(block)
    return blocks

def evaluate_predictions(fname):
    with open(fname, 'rb') as handle:
        results = pickle.load(handle)

    tp = TowerPlanner(stability_mode='contains')
    
    # Index this as [stable][cog_stable][pw_stable]
    for ix, (towers, labels, preds) in enumerate(results):
        correct = [[[0, 0],[0, 0]],[[0, 0],[0, 0]]]
        total = [[[0, 0],[0, 0]],[[0, 0],[0, 0]]]

        # Check the tower stability type.
        for tower, label, pred in zip(towers, labels, preds):
            blocks = to_blocks(tower)

            cog_stable = tp.tower_is_cog_stable(blocks)
            pw_stable = tp.tower_is_constructible(blocks)
            stable = tp.tower_is_stable(blocks)
            if stable != label:
                print('WAT', stable, label)
            #assert stable == label
            total[stable][cog_stable][pw_stable] += 1
            if (pred>0.5) == label:
                correct[stable][cog_stable][pw_stable] += 1

        print(total)
        print('%d Towers' % (ix+2))
        for stable in [0, 1]:
            for cog_stable in [0, 1]:
                for pw_stable in [0, 1]:
                    if ix == 0 and pw_stable != stable:
                        continue
                    acc = correct[stable][cog_stable][pw_stable]/total[stable][cog_stable][pw_stable]
                    print('Stable: %d\tCOG_Stable: %d\tPW_Stable: %d\tAcc: %f' % (stable, cog_stable, pw_stable, acc))


def plot_data_distribution(fname):
    with open(fname, 'rb') as handle:
        data = pickle.load(handle)

    towers = data['5block']['towers']
    towers[:,1:,7:9] -= towers[:,:-1,7:9]

    for tower in towers[::100]:
        plt.scatter(tower[:,7], tower[:,8])

    plt.show()

if __name__ == '__main__':
    # FNAME = 'random_blocks_(x20000)_quat.pkl'
    # dataset = load_dataset(FNAME)
    # check_stability_type(dataset)
    #evaluate_predictions('gat_preds.pkl')
    plot_data_distribution('learning/data/random_blocks_(x10000)_5blocks_all.pkl')
