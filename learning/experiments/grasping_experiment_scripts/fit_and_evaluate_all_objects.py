import argparse
import os


# RUN and CREATE a group file.

# FIT_CMD = 'python -m learning.experiments.active_fit_grasping_pf --exp-name grasp-fit-%s-object%d --objects-fname learning/data/grasping/objects/ycbpowerdrill_10p_set1.pkl --pretrained-ensemble-exp-path learning/experiments/logs/grasping-ycbpowerdrill-latents-m7-20220412-210215/ --ensemble-tx 0 --eval-object-ix %d --strategy %s --max-acquisitions 10 --n-samples %d --n-particles %d' 
# EVAL_CMD = 'python -m learning.evaluate.evaluate_grasping --exp-path %s --val-dataset-fname learning/data/grasping/grasps/ycbpowerdrill_set1_object%d_500g_test.pkl'


FIT_CMD = 'python -m learning.experiments.active_fit_grasping_pf --exp-name grasp-ycbhammer-fit-%s-object%d --objects-fname learning/data/grasping/objects/ycbhammer_10p_set1.pkl --pretrained-ensemble-exp-path learning/experiments/logs/grasping-ycbhammer-latents-m7-20220413-214006/ --ensemble-tx 0 --eval-object-ix %d --strategy %s --max-acquisitions 25 --n-samples %d --n-particles %d' 
EVAL_CMD = 'python -m learning.evaluate.evaluate_grasping --exp-path %s --val-dataset-fname learning/data/grasping/grasps/ycbpowerdrill_set1_object%d_500g_test.pkl'


parser = argparse.ArgumentParser()
parser.add_argument('--strategy', type=str, choices=['bald', 'random'], required=True)
parser.add_argument('--n-samples', type=int, default=100)
parser.add_argument('--n-particles', type=int, default=100)
parser.add_argument('--group-name', type=str, required=True)
args = parser.parse_args()

def get_path(args, ox):
    path_base = 'grasp-ycbhammer-fit-%s-object%d' % (args.group_name, ox)
    for log_dir in os.listdir('learning/experiments/logs'):
        if path_base in log_dir:
            return os.path.join('learning/experiments/logs', log_dir)

if __name__ == '__main__':

    names = []
    object_ids = range(0, 10)
    object_ids = [3, 7]
    for ox in object_ids:
        fit_cmd = FIT_CMD % (args.group_name, ox, ox, args.strategy, args.n_samples, args.n_particles)
        print(fit_cmd)
        os.system(fit_cmd)

        path = get_path(args, ox)
        names.append(path)

        print(eval_cmd)
        eval_cmd = EVAL_CMD % (path, ox)
        os.system(eval_cmd)
    
    with open('learning/evaluate/grasping_run_groups/%s.txt' % args.group_name, 'w') as handle:
        handle.write('\n'.join(names))
        