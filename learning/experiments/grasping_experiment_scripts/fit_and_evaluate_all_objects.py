import argparse
import os
import pickle

# RUN and CREATE a group file.

# FIT_CMD = 'python -m learning.experiments.active_fit_grasping_pf --exp-name grasp-fit-%s-object%d --objects-fname learning/data/grasping/objects/ycbpowerdrill_10p_set1.pkl --pretrained-ensemble-exp-path learning/experiments/logs/grasping-ycbpowerdrill-latents-m7-20220412-210215/ --ensemble-tx 0 --eval-object-ix %d --strategy %s --max-acquisitions 10 --n-samples %d --n-particles %d' 
# EVAL_CMD = 'python -m learning.evaluate.evaluate_grasping --exp-path %s --val-dataset-fname learning/data/grasping/grasps/ycbpowerdrill_set1_object%d_500g_test.pkl'

OBJECTS_FILE = 'learning/data/grasping/ycbsplit1/objects/test_objects.pkl'
FIT_CMD = 'python -m learning.experiments.active_fit_grasping_pf --exp-name grasp-ycbsplit1-fit-%s-object%d --objects-fname learning/data/grasping/ycbsplit1/objects/test_objects.pkl --pretrained-ensemble-exp-path learning/experiments/logs/grasp-ycbsplit1-latents-m5-20220415-121932 --ensemble-tx 0 --eval-object-ix %d --strategy %s --max-acquisitions 10 --n-samples %d --n-particles %d' 
EVAL_CMD = 'python -m learning.evaluate.evaluate_grasping --exp-path %s --val-dataset-fname learning/data/grasping/ycbsplit1/grasps/fitting_phase/fit_grasps_object%d.pkl'

# OBJECTS_FILE = 'learning/data/grasping/ycbsplit1/objects/test_objects_samegeo.pkl'
# FIT_CMD = 'python -m learning.experiments.active_fit_grasping_pf --exp-name grasp-ycbsplit1-fit-%s-object%d --objects-fname learning/data/grasping/ycbsplit1/objects/test_objects_samegeo.pkl --pretrained-ensemble-exp-path learning/experiments/logs/grasp-ycbsplit1-latents-m5-20220415-121932 --ensemble-tx 0 --eval-object-ix %d --strategy %s --max-acquisitions 10 --n-samples %d --n-particles %d' 
# EVAL_CMD = 'python -m learning.evaluate.evaluate_grasping --exp-path %s --val-dataset-fname learning/data/grasping/ycbsplit1/grasps/fitting_phase/fit_grasps_train_object%d.pkl'


parser = argparse.ArgumentParser()
parser.add_argument('--strategy', type=str, choices=['bald', 'random'], required=True)
parser.add_argument('--n-samples', type=int, default=100)
parser.add_argument('--n-particles', type=int, default=100)
parser.add_argument('--group-name', type=str, required=True)
args = parser.parse_args()

def get_path(args, ox):
    path_base = 'grasp-ycbsplit1-fit-%s-object%d' % (args.group_name, ox)
    for log_dir in os.listdir('learning/experiments/logs'):
        if path_base in log_dir:
            return os.path.join('learning/experiments/logs', log_dir)

if __name__ == '__main__':
    with open(OBJECTS_FILE, 'rb') as handle:
        objects = pickle.load(handle)
    n_objects = len(objects['object_data']['object_names'])

    names = []
    object_ids = range(0, n_objects)
    for ox in object_ids:
        print('Fitting object %d/%d...' % (ox, n_objects))
        fit_cmd = FIT_CMD % (args.group_name, ox, ox, args.strategy, args.n_samples, args.n_particles)
        print(fit_cmd)
        os.system(fit_cmd)

        path = get_path(args, ox)
        names.append(path)

        eval_cmd = EVAL_CMD % (path, ox)
        print(eval_cmd)
        os.system(eval_cmd)
    
    with open('learning/evaluate/grasping_run_groups/%s.txt' % args.group_name, 'w') as handle:
        handle.write('\n'.join(names))
        