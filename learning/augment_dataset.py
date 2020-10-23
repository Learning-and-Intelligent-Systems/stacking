import argparse
import numpy as np
import pickle
import os
import time

from block_utils import World, Environment, Dimensions, Object, Quaternion, Pose, Position, ZERO_POS, rotation_group, get_rotated_block
from scipy.spatial.transform import Rotation as R


def augment(all_data, K_skip, translate, vis_tower=False):
    datasets = {}
    for num_blocks in range(2, 6):
        print('Augmenting %d block towers...' % num_blocks)
        data = all_data[f'{num_blocks}block']
        # load the tower data
        towers = data['towers'][::K_skip, :]
        labels = data['labels'][::K_skip]
        N, K, D = towers.shape
        
        if translate:
            N_shift = 4
        else:
            N_shift = 0
        augmented_towers = np.zeros((N*(4+N_shift), K, D))
        augmented_labels = np.zeros((N*(4+N_shift)))
        for ix in range(N):
            if ix % 1000 == 0:
                print(ix)
            tower = [Object.from_vector(towers[ix, jx, :]) for jx in range(num_blocks)]
      
            for kx, z_rot in enumerate([0., np.pi/2., np.pi, 3*np.pi/2]):
                rot = R.from_rotvec([0., 0., z_rot])
                rot_tower = [Object.from_vector(towers[ix, jx, :]) for jx in range(num_blocks)]

                poses = np.array([b.pose.pos for b in rot_tower])
                rot_poses = rot.apply(poses)
                for bx in range(num_blocks):
                    block = rot_tower[bx]
                    new_pose = Pose(Position(*rot_poses[bx,:].tolist()),
                                    Quaternion(*rot.as_quat().tolist()))
                    # new_pose = Pose(Position(*rot.apply(block.pose.pos)),
                    #                 Quaternion(*rot.as_quat().tolist()))
                    block.set_pose(new_pose)
                    rot_tower[bx] = get_rotated_block(block)
                    augmented_towers[(4+N_shift)*ix + kx, bx, :] = rot_tower[bx].vectorize()                    
                augmented_labels[(4+N_shift)*ix + kx] = labels[ix]                   
                
                if translate:
                    dx, dy = np.random.uniform(-0.2, 0.2, 2)
                    shifted_tower = augmented_towers[(4+N_shift)*ix + kx, :].copy()
                    # Indices 7,8 correspond to the pose.
                    # The CoM doesn't need to be shifted because it is relative.
                    shifted_tower[:, 7] += dx
                    shifted_tower[:, 8] += dy
                    
                    augmented_towers[(4+N_shift)*ix + 4 + kx, :, :] = shifted_tower
                    augmented_labels[(4+N_shift)*ix + 4 + kx] = labels[ix]  
                
                if vis_tower:
                    w = World(rot_tower)
                    env = Environment([w], vis_sim=True, vis_frames=True)
                    for tx in range(240):
                        env.step(vis_frames=False)
                        time.sleep(1/240.)
                    env.disconnect()

        datasets[f'{num_blocks}block'] = {'towers': augmented_towers,
                                          'labels': augmented_labels}
    
    return datasets

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str, required=True)
    parser.add_argument('--K', type=int, required=True)
    parser.add_argument('--translate', action='store_true', default=False)
    args = parser.parse_args()
    print(args)
    # 'learning/data/random_blocks_(x2000)_5blocks_uniform_mass.pkl'
   
    with open(args.fname, 'rb') as handle:
        data = pickle.load(handle)
    
    aug_data = augment(data, args.K, args.translate, vis_tower=False)

    # Save the new dataset.
    root, ext = os.path.splitext(args.fname)
    fname = '%s_%daug_%dshift%s' % (root, args.K, args.translate, ext)
    print('Saving to: %s' % fname)
    with open(fname, 'wb') as handle:
        pickle.dump(aug_data, handle)