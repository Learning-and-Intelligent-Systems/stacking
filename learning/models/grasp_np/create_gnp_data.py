import argparse
from email.policy import default
from re import T
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import shutil

from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
from learning.domains.grasping.generate_grasp_datasets import graspablebody_from_vector
from learning.domains.grasping.grasp_data import GraspParallelDataLoader
from pb_robot.planners.antipodalGraspPlanner import GraspSimulationClient, GraspableBody


def transform_points(points, finger1, finger2, ee, viz_data=False):
    midpoint = (finger1 + finger2)/2
    new_x = (finger1 - finger2)/np.linalg.norm(finger1 - finger2)
    new_y = (midpoint - ee)/np.linalg.norm(midpoint - ee)
    new_z = np.cross(new_x, new_y)
    
    # Build transfrom from world frame to grasp frame.
    R, t = np.hstack([new_x[:, None], new_y[:, None], new_z[:, None]]), midpoint
    tform = np.eye(4)
    tform[0:3, 0:3] = R
    tform[0:3, 3] = t
    inv_tform = np.linalg.inv(tform)

    new_points = np.hstack([points, np.ones((points.shape[0], 1))])
    new_points = (inv_tform@new_points.T).T[:, 0:3]

    if viz_data:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')

        ax1.scatter(points[:, 0], points[:, 1], points[:, 2], color='k', alpha=0.2)
        ax1.scatter(*finger1, color='r')
        ax1.scatter(*finger2, color='g')
        ax1.scatter(*ee, color='b')

        ax2.scatter(new_points[:, 0], new_points[:, 1], new_points[:, 2], color='k', alpha=0.2)
        ax2.scatter(0, 0, 0, color='r')

        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')
        
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('z')
        plt.show()

    return new_points.astype('float32')


def process_geometry(train_dataset, radius=0.02, skip=1):
    object_names = train_dataset['object_data']['object_names']
    object_properties = train_dataset['object_data']['object_properties']

    all_grasps = train_dataset['grasp_data']['grasps'][::skip]
    all_ids = train_dataset['grasp_data']['object_ids'][::skip]
    all_labels = train_dataset['grasp_data']['labels'][::skip]
    gx = 0

    # Collect all mesh points for each object.
    all_points_per_objects = {}
    for grasp_vector, object_id in zip(all_grasps, all_ids):
        mesh_points = grasp_vector[3:, 0:3]
        if object_id not in all_points_per_objects:
            all_points_per_objects[object_id] = mesh_points
        else:
            all_points_per_objects[object_id] = np.concatenate([all_points_per_objects[object_id], mesh_points], axis=0)

    # For each grasp, find close points and convert them to the local grasp frame.
    new_geometries_dict = defaultdict(list) # obj_id -> [grasp__points]
    new_midpoints_dict = defaultdict(list) # obj_id -> [grasp_midpoint]
    new_labels_dict = defaultdict(list) # obj_id -> [grasp_label]
    new_meshes_dict = defaultdict(list)

    for grasp_vector, object_id, label in zip(all_grasps, all_ids, all_labels):
        print(f'Coverting grasp {gx}/{len(all_ids)}...')
        gx += 1

        object_name = object_names[object_id]
        object_property = object_properties[object_id]

        graspable_body = graspablebody_from_vector(object_name, object_property)
        
        finger1 = grasp_vector[0, 0:3]
        finger2 = grasp_vector[1, 0:3]
        ee = grasp_vector[2, 0:3]
        midpoint = (finger1 + finger2)/2.0

        # Only sample points that are within 2cm of a grasp point.
        candidate_points = all_points_per_objects[object_id]
        d1 = np.linalg.norm(candidate_points-finger1, axis=1)
        d2 = np.linalg.norm(candidate_points-finger2, axis=1)
        to_keep = np.logical_or(d1 < radius, d2 < radius)
        points = candidate_points[to_keep][:512]
        n_found = points.shape[0]
        print(f'{n_found} points found for grasp {gx}, object {object_id}')

        # Put everything in grasp point ref frame for better generalization. (Including grasp points)
        if gx % 200 == 0:
            viz_data = False
        else:
            viz_data=False
        points = transform_points(points, finger1, finger2, ee, viz_data=viz_data)

        # Assemble dataset.
        new_geometries_dict[object_id].append(points)
        new_midpoints_dict[object_id].append(midpoint)
        new_labels_dict[object_id].append(label)  
        new_meshes_dict[object_id].append(all_points_per_objects[object_id][:512,:])      

    dataset = {
        'grasp_data': {
            'object_meshes': new_meshes_dict,
            'grasp_geometries': new_geometries_dict,
            'grasp_midpoints': new_midpoints_dict,
            'labels': new_labels_dict
        },
        'object_data': train_dataset['object_data'],
        'metadata': train_dataset['metadata']
    }
    return dataset

DATA_ROOT = 'learning/data/grasping'
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dataset-fname', type=str, required=True)
    parser.add_argument('--out-dataset-fname', type=str, required=True)
    args = parser.parse_args()
    print(args)

    in_data_path = os.path.join(DATA_ROOT, args.in_dataset_fname)

    train_data_path = os.path.join(in_data_path, 'grasps', 'training_phase', 'train_grasps.pkl')
    with open(train_data_path, 'rb') as handle:
        train_dataset = pickle.load(handle) 

    new_train_dataset = process_geometry(
        train_dataset,
        radius=0.03,
        skip=1,
    )

    val_data_path = os.path.join(in_data_path, 'grasps', 'training_phase', 'val_grasps.pkl')
    with open(val_data_path, 'rb') as handle:
        val_dataset = pickle.load(handle) 

    new_val_dataset = process_geometry(
        val_dataset,
        radius=0.03,
        skip=1
    )

    in_data_root_path = os.path.join(DATA_ROOT, args.in_dataset_fname)
    out_data_root_path = os.path.join(DATA_ROOT, args.out_dataset_fname)
    os.mkdir(out_data_root_path)

    in_args_path = os.path.join(in_data_root_path, 'args.pkl')
    out_args_path = os.path.join(out_data_root_path, 'args.pkl')
    shutil.copy(in_args_path, out_args_path)
    
    in_objects_path = os.path.join(in_data_root_path, 'objects')
    out_objects_path = os.path.join(out_data_root_path, 'objects')
    os.mkdir(out_objects_path)

    in_train_geo_train_props = os.path.join(in_objects_path, 'train_geo_train_props.pkl')
    in_train_geo_test_props = os.path.join(in_objects_path, 'train_geo_test_props.pkl')
    in_test_geo_test_props = os.path.join(in_objects_path, 'test_geo_test_props.pkl')
    out_train_geo_train_props = os.path.join(out_objects_path, 'train_geo_train_props.pkl')
    out_train_geo_test_props = os.path.join(out_objects_path, 'train_geo_test_props.pkl')
    out_test_geo_test_props = os.path.join(out_objects_path, 'test_geo_test_props.pkl')
    shutil.copy(in_train_geo_train_props, out_train_geo_train_props)
    shutil.copy(in_train_geo_test_props, out_train_geo_test_props)
    shutil.copy(in_test_geo_test_props, out_test_geo_test_props)

    out_grasps_path = os.path.join(out_data_root_path, 'grasps')
    os.mkdir(out_grasps_path)

    out_training_phase_path = os.path.join(out_grasps_path, 'training_phase')
    os.mkdir(out_training_phase_path)

    train_grasps_path = os.path.join(out_training_phase_path, 'train_grasps.pkl') 
    val_grasps_path = os.path.join(out_training_phase_path, 'val_grasps.pkl')
    with open(train_grasps_path, 'wb') as handle:
        pickle.dump(new_train_dataset, handle)
    with open(val_grasps_path, 'wb') as handle:
        pickle.dump(new_val_dataset, handle)
