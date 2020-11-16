import argparse
from copy import deepcopy
import numpy as np
import pickle
import os
import time

from block_utils import World, Environment, Dimensions, Object, Quaternion, Pose, Position, ZERO_POS, rotation_group, get_rotated_block
from scipy.spatial.transform import Rotation as R


def augment(all_data, K_skip, translate=False, mirror=False, vis_tower=False):
    datasets = {}
    for num_blocks in range(2, 6):
        print('Augmenting %d block towers...' % num_blocks)
        data = all_data[f'{num_blocks}block']
        # load the tower data
        towers = data['towers'][::K_skip, :]
        labels = data['labels'][::K_skip]
        N, K, D = towers.shape
        # calculate the number of augmented towers that will be created
        N_angles = 4
        N_shift = 4 if translate else 1
        N_mirror = 3 if mirror else 1
        tower_multiplier = N_angles * N_mirror * N_shift
        N_towers_to_add = N * tower_multiplier
        # and create new arrays to store those towers
        augmented_towers = np.zeros((N_towers_to_add, K, D))
        augmented_labels = np.zeros(N_towers_to_add)

        for ix in range(N):
            if ix % 1000 == 0:
                print(ix)
            original_tower = [Object.from_vector(towers[ix, jx, :]) for jx in range(num_blocks)]

            for kx, z_rot in enumerate([0., np.pi/2., np.pi, 3*np.pi/2]):
                rot = R.from_rotvec([0., 0., z_rot])
                # rotate each block in the tower and add the new tower to the dataset
                rot_poses = rot.apply(np.array([b.pose.pos for b in original_tower]))
                rot_tower = []
                for bx in range(num_blocks):
                    rot_block = deepcopy(original_tower[bx])
                    new_pose = Pose(Position(*rot_poses[bx,:].tolist()),
                                    Quaternion(*rot.as_quat().tolist()))
                    # new_pose = Pose(Position(*rot.apply(block.pose.pos)),
                    #                 Quaternion(*rot.as_quat().tolist()))
                    rot_block.set_pose(new_pose)
                    rot_tower.append(get_rotated_block(rot_block))
                    augmented_towers[tower_multiplier*ix + kx, bx, :] = rot_tower[bx].vectorize()
                augmented_labels[tower_multiplier*ix + kx] = labels[ix]

                # translate the base block in the tower and add after the rotated blocks
                # if translate:
                #     dx, dy = np.random.uniform(-0.2, 0.2, 2)
                #     shifted_tower = augmented_towers[(4+N_shift)*ix + kx, :].copy()
                #     # Indices 7,8 correspond to the pose.
                #     # The CoM doesn't need to be shifted because it is relative.
                #     shifted_tower[:, 7] += dx
                #     shifted_tower[:, 8] += dy

                #     augmented_towers[tower_multiplier*ix + N_angles + kx, :, :] = shifted_tower
                #     augmented_labels[tower_multiplier*ix + N_angles + kx] = labels[ix]

            # flip the mirror the COM about the COG and negate the relative position in x
            # and y for each block. Creates a new tower that is the mirror of the original
            # tower about the x and y axes
            if mirror:
                start_index = tower_multiplier*ix
                # pull out a vector of all the towers we just added. This will be of shape
                # [N_angles x num_blocks x D]
                rot_towers = augmented_towers[start_index: start_index+N_angles, ...]

                # create a vector that will mirror a tower when we multiply it
                # indices 1 and 7 correspond to the x coordinates of COM and relative position
                # indices 2 and 8 correspond to the y coordinates of COM and relative position
                mirror_in_x = np.ones([1,1,D])
                mirror_in_y = np.ones([1,1,D])
                mirror_in_x[..., [1,7]] *= -1
                mirror_in_y[..., [2,8]] *= -1

                # add the mirrored towers to the augmented towers dataset
                augmented_towers[start_index+N_angles*1 : start_index+N_angles*2, ...] = rot_towers * mirror_in_x
                augmented_towers[start_index+N_angles*2 : start_index+N_angles*3, ...] = rot_towers * mirror_in_y
                augmented_labels[start_index:start_index+N_angles*N_mirror] = labels[ix]

            if vis_tower:
                for i in range(tower_multiplier):
                    print('VISUALIZE', ix*tower_multiplier+i, N_towers_to_add)
                    new_tower = [Object.from_vector(augmented_towers[ix*tower_multiplier+i, jx, :]) for jx in range(num_blocks)]
                    w = World(new_tower)
                    env = Environment([w], vis_sim=True, vis_frames=True)
                    for tx in range(60):
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