import argparse
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
        N_shift = 4 if translate else 0
        mirror_multiplier = 4 if mirror else 1
        tower_multiplier = N_angles*mirror_multiplier + N_shift
        N_towers_to_add = N*tower_multiplier
        # and create new arrays to store those towers
        augmented_towers = np.zeros((N_towers_to_add, K, D))
        augmented_labels = np.zeros(N_towers_to_add)
        # NOTE(izzy): the way this is currently written, towers get added to
        # the augmented array in this order for each iteration of this for loop
        # [rot_1, rot_2, rot_3, rot_4,
        #  shift_1, shift_2, shift_3, shift_4,
        #  mirror_1_x, mirror_1_y, mirror_1_xy,
        #  mirror_2_x, mirror_2_y, mirror_2_xy,
        #  mirror_3_x, mirror_3_y, mirror_3_xy,
        #  mirror_4_x, mirror_4_y, mirror_4_xy]
        for ix in range(N):
            if ix % 1000 == 0:
                print(ix)
            tower = [Object.from_vector(towers[ix, jx, :]) for jx in range(num_blocks)]

            for kx, z_rot in enumerate([0., np.pi/2., np.pi, 3*np.pi/2]):
                rot = R.from_rotvec([0., 0., z_rot])
                rot_tower = [Object.from_vector(towers[ix, jx, :]) for jx in range(num_blocks)]

                # rotate each block in the tower and add the new tower to the dataset
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
                    augmented_towers[tower_multiplier*ix + kx, bx, :] = rot_tower[bx].vectorize()
                augmented_labels[tower_multiplier*ix + kx] = labels[ix]

                # translate the base block in the tower and add after the rotated blocks
                if translate:
                    dx, dy = np.random.uniform(-0.2, 0.2, 2)
                    shifted_tower = augmented_towers[(4+N_shift)*ix + kx, :].copy()
                    # Indices 7,8 correspond to the pose.
                    # The CoM doesn't need to be shifted because it is relative.
                    shifted_tower[:, 7] += dx
                    shifted_tower[:, 8] += dy

                    augmented_towers[tower_multiplier*ix + N_angles + kx, :, :] = shifted_tower
                    augmented_labels[tower_multiplier*ix + N_angles + kx] = labels[ix]

                # flip the mirror the COM about the COG and negate the relative position in x
                # and y for each block. Creates a new tower that is the mirror of the original
                # tower about the x and y axes
                if mirror:
                    mirrored_in_x_tower = augmented_towers[tower_multiplier*ix + kx, :].copy()
                    mirrored_in_y_tower = augmented_towers[tower_multiplier*ix + kx, :].copy()
                    mirrored_in_xy_tower = augmented_towers[tower_multiplier*ix + kx, :].copy()
                    # indices 1 and 7 correspond to the x coordinates of COM and relative position
                    # indices 2 and 8 correspond to the y coordinates of COM and relative position
                    mirrored_in_x_tower[:, [1,7]] *= -1
                    mirrored_in_y_tower[:, [2,8]] *= -1
                    mirrored_in_xy_tower[:, [1,2,7,8]] *= -1
                    # add the mirrored towers to the augmented towers dataset
                    start_index = tower_multiplier*ix + N_angles + N_shift + kx*3
                    augmented_towers[start_index+0, :, :] = mirrored_in_x_tower
                    augmented_towers[start_index+1, :, :] = mirrored_in_y_tower
                    augmented_towers[start_index+2, :, :] = mirrored_in_xy_tower
                    augmented_labels[start_index:start_index+3] = labels[ix]

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