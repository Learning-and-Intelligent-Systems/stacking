"""
Each experiment involves many dataset artifacts. This involves one large training
dataset (and corresponding validation set using the same objects), and many smaller
adaptation datasets used to evaluate accuracy throughout adaptation.

Three object sets are created:
- train_geo_train_props
- train_geo_test_props
- test_geo_test_pros

Fitting phase datasets are created for each of these to cover a range of difficulty:
- fit_grasps_train_geo_trainprop (easy: test on same objects used to train)
- fit_grasps_train_geo (medium: test on same geometry but different intrinsic properties)
- fit_grasps_test_geo (hard: test on different geometry and parts)
"""

import argparse
import multiprocessing
import pickle
import os

from types import SimpleNamespace

from learning.domains.grasping.generate_grasp_datasets import generate_objects, generate_datasets


def get_object_list(fname):
    """ Parse object text file. """
    print('Opening', fname)
    with open(fname, 'r') as handle:
        objects = handle.readlines()
    return [o.strip() for o in objects if len(o) > 1]

def parse_ignore_file(fname):
    """ Sometimes we can't grasp objects due to geometry. Specify object to skip in an ignore.txt file. """
    with open(fname, 'r') as handle:
        lines = [l.strip().split(',') for l in handle.readlines()]

    train_skip, test_skip = [], []
    for split, ox in lines:
        if split == 'train':
            train_skip.append(int(ox))
        else:
            test_skip.append(int(ox))

    return train_skip, test_skip

def merge_datasets(dataset_paths, merged_fname):
    """ Create one large dataset file form individual object files."""
    all_grasps, all_object_ids, all_labels = [], [], []
    for dataset_path in dataset_paths:
        print('Loading:', dataset_path)
        if not os.path.exists(dataset_path):
            continue

        with open(dataset_path, 'rb') as handle:
            single_dataset = pickle.load(handle)

        all_grasps += [g.astype('float32') for g in single_dataset['grasp_data']['grasps']]
        all_object_ids += single_dataset['grasp_data']['object_ids']
        all_labels += single_dataset['grasp_data']['labels']

    updated_metadata = single_dataset['metadata']
    updated_metadata.fname = merged_fname
    updated_metadata.object_ix = -1

    merged_dataset = {
        'grasp_data': {
            'grasps': all_grasps,
            'object_ids': all_object_ids,
            'labels': all_labels
        },
        'object_data': single_dataset['object_data'],
        'metadata': updated_metadata
    }

    with open(merged_fname, 'wb') as handle:
        pickle.dump(merged_dataset, handle)


parser = argparse.ArgumentParser()
parser.add_argument('--train-objects-fname', type=str, required=True)
parser.add_argument('--test-objects-fname', type=str, required=True)
parser.add_argument('--data-root-name', type=str, required=True)
parser.add_argument('--n-property-samples-train', type=int, required=True)
parser.add_argument('--n-property-samples-test', type=int, required=True)
parser.add_argument('--n-grasps-per-object', type=int, required=True)
parser.add_argument('--n-points-per-object', type=int, required=True)
parser.add_argument('--n-fit-grasps', type=int, required=True)
parser.add_argument('--grasp-noise', type=float, required=True)
parser.add_argument('--n-processes', type=int, default=1)
new_args = parser.parse_args()
print(new_args)


if __name__ == '__main__':

    # Directory setup.
    data_root_path = os.path.join('learning', 'data', 'grasping', new_args.data_root_name)
    args_path = os.path.join(data_root_path, 'args.pkl')
    if os.path.exists(data_root_path):
        input('[Warning] Data root directory already exists. Continue using initial args?')
        with open(args_path, 'rb') as handle:
            args = pickle.load(handle)
            args.n_processes = new_args.n_processes
    else:
        os.mkdir(data_root_path)
        with open(args_path, 'wb') as handle:
            pickle.dump(new_args, handle)
        args = new_args

    worker_pool = multiprocessing.Pool(
        processes=args.n_processes,
        maxtasksperchild=1
    )

    objects_path = os.path.join(data_root_path, 'objects')
    grasps_path = os.path.join(data_root_path, 'grasps')
    if not os.path.exists(objects_path):
        os.mkdir(objects_path)
        os.mkdir(grasps_path)

    train_objects = get_object_list(args.train_objects_fname)
    test_objects = get_object_list(args.test_objects_fname)

    # Generate initial object sets.
    object_tasks = []

    print('[Objects] Generating train objects.')
    train_objects_path = os.path.join(objects_path, 'train_geo_train_props.pkl')
    if not os.path.exists(train_objects_path):
        train_objects_args = SimpleNamespace(
            fname=train_objects_path,
            object_names=train_objects,
            n_property_samples=args.n_property_samples_train)
        object_tasks.append(train_objects_args)

    print('[Objects] Generating test objects: novel geometry.')
    test_objects_path = os.path.join(objects_path, 'test_geo_test_props.pkl')
    if not os.path.exists(test_objects_path):
        test_objects_args = SimpleNamespace(
            fname=test_objects_path,
            object_names=test_objects,
            n_property_samples=args.n_property_samples_test)
        object_tasks.append(test_objects_args)

    print('[Objects] Generating test objects: train geometry.')
    test_objects_samegeo_path = os.path.join(objects_path, 'train_geo_test_props.pkl')
    if not os.path.exists(test_objects_samegeo_path):
        test_objects_samegeo_args = SimpleNamespace(
            fname=test_objects_samegeo_path,
            object_names=train_objects,
            n_property_samples=args.n_property_samples_test)
        object_tasks.append(test_objects_samegeo_args)

    worker_pool.map(generate_objects, object_tasks)

    # Generate training and validation sets used for the training phase.
    training_phase_path = os.path.join(grasps_path, 'training_phase')
    if not os.path.exists(training_phase_path):
        os.mkdir(training_phase_path)

    TRAIN_IGNORE, TEST_IGNORE = parse_ignore_file(os.path.join(data_root_path, 'ignore.txt'))
    train_dataset_tasks, train_dataset_paths = [], []

    print('[Grasps] Generating train grasps for training phase.')
    for ox in range(0, len(train_objects)*args.n_property_samples_train):
        if ox in TRAIN_IGNORE:
            continue
        train_grasps_path = os.path.join(training_phase_path, f'train_grasps_object{ox}.pkl')
        train_dataset_paths.append(train_grasps_path)
        if not os.path.exists(train_grasps_path):
            train_grasps_args = SimpleNamespace(
                fname=train_grasps_path,
                objects_fname=train_objects_path,
                n_points_per_object=args.n_points_per_object,
                n_grasps_per_object=args.n_grasps_per_object,
                object_ix=ox,
                grasp_noise=args.grasp_noise)
            train_dataset_tasks.append(train_grasps_args)

    worker_pool.map(generate_datasets, train_dataset_tasks)

    print('[Grasps] Merging all train grasps for training phase.')
    train_grasps_path = os.path.join(training_phase_path, 'train_grasps.pkl')
    merge_datasets(train_dataset_paths, train_grasps_path)

    print('[Grasps] Generating validation grasps for training phase.')
    val_dataset_tasks, val_dataset_paths = [], []
    for ox in range(0, len(train_objects)*args.n_property_samples_train):
        if ox in TRAIN_IGNORE:
            continue
        val_grasps_path = os.path.join(training_phase_path, f'val_grasps_object{ox}.pkl')
        if not os.path.exists(val_grasps_path):
            val_grasps_args = SimpleNamespace(
                fname=val_grasps_path,
                objects_fname=train_objects_path,
                n_points_per_object=args.n_points_per_object,
                n_grasps_per_object=10,
                object_ix=ox,
                grasp_noise=args.grasp_noise)
            val_dataset_tasks.append(val_grasps_args)

    worker_pool.map(generate_datasets, val_dataset_tasks)

    print('[Grasps] Merging all validation grasps for training phase.')
    val_grasps_path = os.path.join(training_phase_path, 'val_grasps.pkl')
    merge_datasets(val_dataset_paths, val_grasps_path)

    # Generate fitting object datasets.
    fitting_phase_path = os.path.join(grasps_path, 'fitting_phase')
    if not os.path.exists(fitting_phase_path):
        os.mkdir(fitting_phase_path)

    fit_dataset_tasks = []
    for ox in range(0, min(100, len(test_objects)*args.n_property_samples_test)):
        if ox in TEST_IGNORE:
            continue
        print(f'[Grasps] Generating grasps for fitting phase eval for obj {ox}.')
        fit_grasps_path = os.path.join(fitting_phase_path, f'fit_grasps_test_geo_object{ox}.pkl')
        if not os.path.exists(fit_grasps_path):
            fit_grasps_args = SimpleNamespace(
                fname=fit_grasps_path,
                objects_fname=test_objects_path,
                n_points_per_object=args.n_points_per_object,
                n_grasps_per_object=args.n_fit_grasps,
                object_ix=ox,
                grasp_noise=args.grasp_noise)
            fit_dataset_tasks.append(fit_grasps_args)

    worker_pool.map(generate_datasets, fit_dataset_tasks)

    fit_dataset_samegeo_tasks = []
    for ox in range(0, min(100, len(train_objects)*args.n_property_samples_test)):
        if ox in TRAIN_IGNORE:
            continue
        print(f'[Grasps] Generating grasps for fitting phase eval for samegeo obj {ox}.')
        fit_grasps_samegeo_path = os.path.join(fitting_phase_path,
                                               f'fit_grasps_train_geo_object{ox}.pkl')
        if not os.path.exists(fit_grasps_samegeo_path):
            fit_grasps_samegeo_args = SimpleNamespace(
                fname=fit_grasps_samegeo_path,
                objects_fname=test_objects_samegeo_path,
                n_points_per_object=args.n_points_per_object,
                n_grasps_per_object=args.n_fit_grasps,
                object_ix=ox,
                grasp_noise=args.grasp_noise)
            fit_dataset_samegeo_tasks.append(fit_grasps_samegeo_args)

    worker_pool.map(generate_datasets, fit_dataset_samegeo_tasks)

    fit_dataset_samegeo_tasks = []
    for ox in range(0, min(100, len(train_objects)*args.n_property_samples_train)):
        if ox in TRAIN_IGNORE:
            continue
        print(f'[Grasps] Generating grasps for fitting phase eval for samegeo sameprop obj {ox}.')
        fit_grasps_samegeo_sameprop_path = os.path.join(
            fitting_phase_path,
            f'fit_grasps_train_geo_trainprop_object{ox}.pkl'
        )
        if not os.path.exists(fit_grasps_samegeo_sameprop_path):
            fit_grasps_samegeo_args = SimpleNamespace(
                fname=fit_grasps_samegeo_sameprop_path,
                objects_fname=train_objects_path,
                n_points_per_object=args.n_points_per_object,
                n_grasps_per_object=args.n_fit_grasps,
                object_ix=ox,
                grasp_noise=0.0)
            fit_dataset_samegeo_tasks.append(fit_grasps_samegeo_args)

    worker_pool.map(generate_datasets, fit_dataset_samegeo_tasks)
