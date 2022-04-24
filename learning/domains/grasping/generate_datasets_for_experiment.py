import argparse
import pickle
import sys
import os

from types import SimpleNamespace

from learning.domains.grasping.generate_grasp_datasets import generate_objects, generate_datasets


parser = argparse.ArgumentParser()
parser.add_argument('--train-objects', nargs='+', help='Either "all" or a list of Ycb object names to include')
parser.add_argument('--test-objects', nargs='+', help='Either "all" or a list of Ycb object names to include')
parser.add_argument('--data-root-name', type=str, required=True)
parser.add_argument('--n-property-samples-train', type=int, required=True)
parser.add_argument('--n-property-samples-test', type=int, required=True)
parser.add_argument('--n-grasps-per-object', type=int, required=True)
parser.add_argument('--n-points-per-object', type=int, required=True)
parser.add_argument('--n-fit-grasps', type=int, required=True)
args = parser.parse_args()


if __name__ == '__main__':
    # Directory setup.
    data_root_path = os.path.join('learning', 'data', 'grasping', args.data_root_name)
    args_path = os.path.join(data_root_path, 'args.pkl')
    if os.path.exists(data_root_path):
        input('[Warning] Data root directory already exists. Continue using initial args?')
        with open(args_path, 'rb') as handle:
            args = pickle.load(handle)
    else:
        os.mkdir(data_root_path)
        with open(args_path, 'wb') as handle:
            pickle.dump(args, handle)
    
    objects_path = os.path.join(data_root_path, 'objects')
    grasps_path = os.path.join(data_root_path, 'grasps')
    if not os.path.exists(objects_path):
        os.mkdir(objects_path)
        os.mkdir(grasps_path)

    # Generate initial object sets.
    print('[Objects] Generating train objects.')
    train_objects_path = os.path.join(objects_path, 'train_objects.pkl')
    if not os.path.exists(train_objects_path):
        train_objects_args = SimpleNamespace(
            fname=train_objects_path, 
            ycb_objects=args.train_objects, 
            n_property_samples=args.n_property_samples_train)
        generate_objects(train_objects_args)
    
    print('[Objects] Generating test objects: novel geometry.')
    test_objects_path = os.path.join(objects_path, 'test_objects.pkl')
    if not os.path.exists(test_objects_path):
        test_objects_args = SimpleNamespace(
            fname=test_objects_path, 
            ycb_objects=args.test_objects, 
            n_property_samples=args.n_property_samples_test)
        generate_objects(test_objects_args)

    print('[Objects] Generating test objects: train geometry.')
    test_objects_samegeo_path = os.path.join(objects_path, 'test_objects_samegeo.pkl')
    if not os.path.exists(test_objects_samegeo_path):
        test_objects_samegeo_args = SimpleNamespace(
            fname=test_objects_samegeo_path, 
            ycb_objects=args.train_objects, 
            n_property_samples=args.n_property_samples_test)
        generate_objects(test_objects_samegeo_args)

    # Generate training and validation sets used for the training phase.
    training_phase_path = os.path.join(grasps_path, 'training_phase')
    if not os.path.exists(training_phase_path):
        os.mkdir(training_phase_path)
    
        print('[Grasps] Generating train grasps for training phase.')
        train_grasps_path = os.path.join(training_phase_path, 'train_grasps.pkl') 
        train_grasps_args = SimpleNamespace(
            fname=train_grasps_path,
            objects_fname=train_objects_path,
            n_points_per_object=args.n_points_per_object,
            n_grasps_per_object=args.n_grasps_per_object,
            object_ix=-1)
        generate_datasets(train_grasps_args)

        print('[Grasps] Generating validation grasps for training phase.')
        val_grasps_path = os.path.join(training_phase_path, 'val_grasps.pkl')
        val_grasps_args = SimpleNamespace(
            fname=val_grasps_path,
            objects_fname=train_objects_path,
            n_points_per_object=args.n_points_per_object,
            n_grasps_per_object=args.n_grasps_per_object,
            object_ix=-1)
        generate_datasets(val_grasps_args)

    # Generate fitting object datasets.
    fitting_phase_path = os.path.join(grasps_path, 'fitting_phase')
    if not os.path.exists(fitting_phase_path):
        os.mkdir(fitting_phase_path)
    
    for ox in range(0, len(args.test_objects*args.n_property_samples_test)):
        print('[Grasps] Generating grasps for evaluating fitting phase for object %d.' % ox)
        fit_grasps_path = os.path.join(fitting_phase_path, 'fit_grasps_object%d.pkl' % ox)
        if not os.path.exists(fit_grasps_path):
            fit_grasps_args = SimpleNamespace(
                fname=fit_grasps_path,
                objects_fname=test_objects_path,
                n_points_per_object=args.n_points_per_object,
                n_grasps_per_object=args.n_fit_grasps,
                object_ix=ox)
            generate_datasets(fit_grasps_args)

    for ox in range(0, len(args.train_objects*args.n_property_samples_test)):
        print('[Grasps] Generating grasps for evaluating fitting phase for samegeo object %d.' % ox)
        fit_grasps_samegeo_path = os.path.join(fitting_phase_path, 'fit_grasps_train_object%d.pkl' % ox)
        if not os.path.exists(fit_grasps_samegeo_path):
            fit_grasps_samegeo_args = SimpleNamespace(
                fname=fit_grasps_samegeo_path,
                objects_fname=test_objects_samegeo_path,
                n_points_per_object=args.n_points_per_object,
                n_grasps_per_object=args.n_fit_grasps,
                object_ix=ox)
            generate_datasets(fit_grasps_samegeo_args)
        


    