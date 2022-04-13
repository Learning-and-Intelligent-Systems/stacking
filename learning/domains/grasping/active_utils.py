import pickle

from learning.active.utils import ActiveExperimentLogger
from learning.domains.grasping.generate_grasp_datasets import graspablebody_from_vector, sample_grasp_X
from pb_robot.planners.antipodalGraspPlanner import GraspSampler, GraspStabilityChecker


def get_train_and_fit_objects(pretrained_ensemble_path, use_latents, fit_objects_fname, fit_object_ix):
    train_logger = ActiveExperimentLogger(exp_path=pretrained_ensemble_path)
    with open(train_logger.args.train_dataset_fname, 'rb') as handle:
        train_dataset = pickle.load(handle)
    
    train_objects = train_dataset['object_data']
    with open(fit_objects_fname, 'rb') as handle:
        fit_objects = pickle.load(handle)['object_data']

    fit_object_name = fit_objects['object_names'][fit_object_ix]
    fit_object_props = fit_objects['object_properties'][fit_object_ix]

    train_objects['object_names'].append(fit_object_name)
    train_objects['object_properties'].append(fit_object_props)
    
    return train_objects


def get_fit_object(object_set):
    name = object_set['object_names'][-1]
    props = object_set['object_properties'][-1]
    ix = len(object_set['object_names']) - 1
    return name, props, ix


def sample_unlabeled_data(n_samples, object_set):
    object_name, object_properties, object_ix = get_fit_object(object_set)
    graspable_body = graspablebody_from_vector(object_name, object_properties)

    object_grasp_data, object_grasp_ids, object_grasp_labels = [], [], []  
    raw_grasps = []  
    for nx in range(n_samples):
        grasp, X = sample_grasp_X(graspable_body, object_properties, n_points_per_object=512)
        
        raw_grasps.append(grasp)
        object_grasp_data.append(X)
        object_grasp_ids.append(object_ix)
        object_grasp_labels.append(0)

    unlabeled_dataset = {
        'grasp_data': {
            'raw_grasps': raw_grasps,
            'grasps': object_grasp_data,
            'object_ids': object_grasp_ids,
            'labels': object_grasp_labels
        },
        'object_data': object_set,
        'metadata': {
            'n_samples': n_samples,
        }
    }
    return unlabeled_dataset


def get_labels(grasp_dataset):
    labeler = GraspStabilityChecker(stability_direction='all', label_type='relpose')
    raw_grasps = grasp_dataset['grasp_data']['raw_grasps']
    for gx in range(0, len(raw_grasps)):
        label = labeler.get_label(raw_grasps[gx], show_pybullet=False)
        print('Label', label)
        grasp_dataset['grasp_data']['labels'][gx] = label
    return grasp_dataset

