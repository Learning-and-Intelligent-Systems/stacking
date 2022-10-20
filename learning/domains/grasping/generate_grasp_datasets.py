import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np

from pb_robot.planners.antipodalGraspPlanner import (
    GraspableBodySampler,
    GraspSampler,
    GraspSimulationClient,
    GraspableBody,
    GraspStabilityChecker
)


def vector_from_graspablebody(graspable_body):
    """ Represent the intrinsic parameters of a body as a vector. """
    vector = np.array(graspable_body.com + (graspable_body.mass, graspable_body.friction))
    return vector

def graspablebody_from_vector(object_name, vector):
    """ Instatiate a GraspableBody given a vector of parameters. """
    graspable_body = GraspableBody(object_name=object_name,
                                   com=tuple(vector[0:3]),
                                   mass=vector[3],
                                   friction=vector[4])
    return graspable_body

def sample_grasp_X(graspable_body, property_vector, n_points_per_object, grasp=None):
    # Sample new point cloud for object.
    sim_client = GraspSimulationClient(graspable_body, False, 'object_models')
    mesh_points = np.array(sim_client.mesh.sample(n_points_per_object, return_index=False),
                           dtype='float32')
    mesh_points = np.hstack([mesh_points,
                             np.ones((n_points_per_object, 1), dtype='float32')])
    mesh_points = (sim_client.mesh_tform@(mesh_points.T)).T[:, 0:3]
    sim_client.disconnect()

    # Sample grasp.
    if grasp is None:
        grasp_sampler = GraspSampler(graspable_body=graspable_body,
                                     antipodal_tolerance=30,
                                     show_pybullet=False)
        grasp = grasp_sampler.sample_grasp(force=20, show_trimesh=False)
        grasp_sampler.disconnect()

    # Encode grasp as points.
    grasp_points = (grasp.pb_point1, grasp.pb_point2, grasp.ee_relpose[0])

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(mesh_points[:, 0], mesh_points[:, 1], mesh_points[:, 2], color='b')
    # grasp_xs = [p[0] for p in grasp_points]
    # grasp_ys = [p[1] for p in grasp_points]
    # grasp_zs = [p[2] for p in grasp_points]
    # ax.scatter(grasp_xs, grasp_ys, grasp_zs, color='r')
    # bound = 0.1
    # ax.set_xlim(-bound, bound)
    # ax.set_ylim(-bound, bound)
    # ax.set_zlim(-bound, bound)
    # plt.savefig('test.png')

    # Add grasp/object indicator features.
    grasp_vectors = np.array(grasp_points, dtype='float32')
    grasp_vectors = np.hstack([grasp_vectors, np.eye(3, dtype='float32')])
    mesh_vectors = np.hstack([mesh_points, np.zeros((n_points_per_object, 3), dtype='float32')])

    # Concatenate all relevant vectors (points, indicators, properties).
    X = np.vstack([grasp_vectors, mesh_vectors])
    props = np.broadcast_to(property_vector, (X.shape[0], len(property_vector)))
    X = np.hstack([X, props])

    return grasp, X

def generate_datasets(dataset_args):
    """ Generate a dataset of grasps and labels for a given object file. """

    assert 'learning/data/grasping' in dataset_args.fname
    with open(dataset_args.objects_fname, 'rb') as handle:
        object_data = pickle.load(handle)['object_data']

    object_grasp_data, object_grasp_ids, object_grasp_labels = [], [], []

    for prop_ix in range(len(object_data['object_names'])):
        if dataset_args.object_ix > -1 and dataset_args.object_ix != prop_ix:
            continue
        print('Property sample %d/%d...' % (prop_ix, len(object_data['object_names'])))

        object_id = prop_ix
        object_name = object_data['object_names'][prop_ix]
        property_vector = object_data['object_properties'][prop_ix]

        graspable_body = graspablebody_from_vector(object_name, property_vector)
        print(f'Object name: {object_name}', property_vector)
        # Sample random grasps with labels.

        labeler = GraspStabilityChecker(graspable_body,
                                        stability_direction='all',
                                        label_type='relpose',
                                        grasp_noise=dataset_args.grasp_noise,
                                        show_pybullet=False)
        print('PBID:', labeler.sim_client.pb_client_id)
        for grasp_ix in range(0, dataset_args.n_grasps_per_object):
            print('Grasp %d/%d...' % (grasp_ix, dataset_args.n_grasps_per_object))

            grasp, X = sample_grasp_X(graspable_body,
                                      property_vector,
                                      dataset_args.n_points_per_object)

            # Get label.
            label = labeler.get_label(grasp)

            object_grasp_data.append(X)
            object_grasp_ids.append(object_id)
            object_grasp_labels.append(int(label))

        labeler.disconnect()

        dataset = {
            'grasp_data': {
                'grasps': object_grasp_data,
                'object_ids': object_grasp_ids,
                'labels': object_grasp_labels
            },
            'object_data': object_data,
            'metadata': dataset_args
        }

    with open('%s' % dataset_args.fname, 'wb') as handle:
        pickle.dump(dataset, handle)


def generate_objects(obj_args):
    """ Sample random objects and intrinsic properties for those objects. """
    assert 'learning/data/grasping' in obj_args.fname
    object_instance_names = []
    object_instance_properties = []
    for obj_ix, name in enumerate(obj_args.object_names):
        print('Object %d/%d...' % (obj_ix, len(obj_args.object_names)))
        for prop_ix in range(0, obj_args.n_property_samples):
            print('Property sample %d/%d...' % (prop_ix, obj_args.n_property_samples))

            object_id = obj_ix*len(obj_args.object_names) + prop_ix

            # Sample a new object.
            graspable_body = GraspableBodySampler.sample_random_object_properties(name)

            # Get property label vector.
            property_vector = vector_from_graspablebody(graspable_body)

            object_instance_names.append(name)
            object_instance_properties.append(property_vector)

    dataset = {
        'object_data': {
            'object_names': object_instance_names,
            'object_properties': object_instance_properties,
            'property_names': ['com_x', 'com_y', 'com_z', 'mass', 'friction']
        },
        'metadata': obj_args
    }
    with open('%s' % obj_args.fname, 'wb') as handle:
        pickle.dump(dataset, handle)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',
                        required=True,
                        choices=['objects', 'grasps'])
    parser.add_argument('--fname',
                        type=str,
                        required=True,
                        help='Base name used for saving all dataset files.')
    # args.mode == 'objects' parameters
    parser.add_argument('--object-names',
                        default=[],
                        nargs='+',
                        help='List of all SN or YCB objects to include.')
    parser.add_argument('--n-property-samples',
                        type=int,
                        default=-1,
                        help='Number of object instances per each YcbObject geometry type.')
    # args.mode == 'grasps' parameters
    parser.add_argument('--objects-fname',
                        default='',
                        type=str,
                        help='File with data about objects and object properties.')
    parser.add_argument('--n-points-per-object',
                        type=int,
                        default=-1,
                        help='Number of points to include in each point cloud.')
    parser.add_argument('--n-grasps-per-object',
                        type=int,
                        default=-1,
                        help='Number of grasps to label per object.')
    parser.add_argument('--object-ix',
                        type=int,
                        default=-1,
                        help='Only use the given object in the dataset.')
    parser.add_argument('--grasp-noise',
                        type=float,
                        default=0.0)
    args = parser.parse_args()
    print(args)

    if args.mode == 'objects':
        assert len(args.object_names) > 0
        assert args.n_property_samples > 0
        generate_objects(args)
    elif args.mode == 'grasps':
        assert len(args.objects_fname) > 0
        assert args.n_points_per_object > 0
        assert args.n_grasps_per_object > 0
        generate_datasets(args)
