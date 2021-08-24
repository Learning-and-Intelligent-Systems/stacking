import argparse
import matplotlib.pyplot as plt
import numpy as np

from learning.active.utils import GoalConditionedExperimentLogger, potential_actions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',
                        action='store_true',
                        help='set to run in debug mode')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

### Parameters
# random-goals-opt (with exploration for unsolved goals and deeper network)
random_goals_opt_paths = ['learning/experiments/logs/datasets/test-random-goals-opt-20210823-165549',
                    'learning/experiments/logs/datasets/test-random-goals-opt-20210823-165810',
                    'learning/experiments/logs/datasets/test-random-goals-opt-20210823-170034',
                    'learning/experiments/logs/datasets/test-random-goals-opt-20210823-170322',
                    'learning/experiments/logs/datasets/test-random-goals-opt-20210823-170613',
                    'learning/experiments/logs/datasets/test-random-goals-opt-20210823-171415',
                    'learning/experiments/logs/datasets/test-random-goals-opt-20210823-174213',
                    'learning/experiments/logs/datasets/test-random-goals-opt-20210823-174438',
                    'learning/experiments/logs/datasets/test-random-goals-opt-20210823-174637',
                    'learning/experiments/logs/datasets/test-random-goals-opt-20210823-174845',
                    'learning/experiments/logs/datasets/test-random-goals-opt-20210823-175114',
                    'learning/experiments/logs/datasets/test-random-goals-opt-20210823-175401',
                    'learning/experiments/logs/datasets/test-random-goals-opt-20210823-175731',
                    'learning/experiments/logs/datasets/test-random-goals-opt-20210823-175921',
                    'learning/experiments/logs/datasets/test-random-goals-opt-20210823-180142',
                    'learning/experiments/logs/datasets/test-random-goals-opt-20210823-180404',
                    'learning/experiments/logs/datasets/test-random-goals-opt-20210823-180615',
                    'learning/experiments/logs/datasets/test-random-goals-opt-20210823-180734',
                    'learning/experiments/logs/datasets/test-random-goals-opt-20210823-181009',
                    'learning/experiments/logs/datasets/test-random-goals-opt-20210823-181216']

# random-goals-learned (with exploration for unsolved goals and deeper network)
random_goals_learned_paths = ['learning/experiments/logs/datasets/test-random-goals-learned-20210823-165618',
                    'learning/experiments/logs/datasets/test-random-goals-learned-20210823-170031',
                    'learning/experiments/logs/datasets/test-random-goals-learned-20210823-170422',
                    'learning/experiments/logs/datasets/test-random-goals-learned-20210823-171341',
                    'learning/experiments/logs/datasets/test-random-goals-learned-20210823-174239',
                    'learning/experiments/logs/datasets/test-random-goals-learned-20210823-174658',
                    'learning/experiments/logs/datasets/test-random-goals-learned-20210823-175109',
                    'learning/experiments/logs/datasets/test-random-goals-learned-20210823-175726',
                    'learning/experiments/logs/datasets/test-random-goals-learned-20210823-180152',
                    'learning/experiments/logs/datasets/test-random-goals-learned-20210823-180630',
                    'learning/experiments/logs/datasets/test-random-goals-learned-20210823-181136',
                    'learning/experiments/logs/datasets/test-random-goals-learned-20210823-181550',
                    'learning/experiments/logs/datasets/test-random-goals-learned-20210823-181840',
                    'learning/experiments/logs/datasets/test-random-goals-learned-20210823-183900',
                    'learning/experiments/logs/datasets/test-random-goals-learned-20210823-185941',
                    'learning/experiments/logs/datasets/test-random-goals-learned-20210823-190252',
                    'learning/experiments/logs/datasets/test-random-goals-learned-20210823-190613',
                    'learning/experiments/logs/datasets/test-random-goals-learned-20210823-191022',
                    'learning/experiments/logs/datasets/test-random-goals-learned-20210823-191356',
                    'learning/experiments/logs/datasets/test-random-goals-learned-20210823-191700']

# random-actions
random_actions_paths = ['learning/experiments/logs/datasets/random-actions-100-20210818-224843',
                    'learning/experiments/logs/datasets/random-actions-100-20210818-224843_1',
                    'learning/experiments/logs/datasets/random-actions-100-20210818-224843_2',
                    'learning/experiments/logs/datasets/random-actions-100-20210818-224843_3',
                    'learning/experiments/logs/datasets/random-actions-100-20210818-224843_4',
                    'learning/experiments/logs/datasets/random-actions-100-20210818-224843_5',
                    'learning/experiments/logs/datasets/random-actions-100-20210818-224843_6',
                    'learning/experiments/logs/datasets/random-actions-100-20210818-224843_7',
                    'learning/experiments/logs/datasets/random-actions-100-20210818-224843_8',
                    'learning/experiments/logs/datasets/random-actions-100-20210818-224844',
                    'learning/experiments/logs/datasets/random-actions-100-20210818-224844_1',
                    'learning/experiments/logs/datasets/random-actions-100-20210818-224844_2',
                    'learning/experiments/logs/datasets/random-actions-100-20210818-224844_3',
                    'learning/experiments/logs/datasets/random-actions-100-20210818-224844_4',
                    'learning/experiments/logs/datasets/random-actions-100-20210818-224844_5',
                    'learning/experiments/logs/datasets/random-actions-100-20210818-224844_6',
                    'learning/experiments/logs/datasets/random-actions-100-20210818-224844_7',
                    'learning/experiments/logs/datasets/random-actions-100-20210818-224844_8',
                    'learning/experiments/logs/datasets/random-actions-100-20210818-224845',
                    'learning/experiments/logs/datasets/random-actions-100-20210818-224845_1']

dataset_exp_paths = random_actions_paths

###
labels = {0: [], 1: []}
heights = {}

# NOTE: this only works if all data_exp_paths have the same num blocks:
dataset_logger = GoalConditionedExperimentLogger(dataset_exp_paths[0])
num_blocks = dataset_logger.load_args().num_blocks

# init keys for all potential keys
pos_actions, neg_actions = potential_actions(num_blocks)
pos_actions = {pos_a : [] for pos_a in pos_actions}
neg_actions = {neg_a : [] for neg_a in neg_actions}

for th in range(1,num_blocks+1): heights[th] = []

# store values for each dataset
for dataset_exp_path in dataset_exp_paths:
    dataset_logger = GoalConditionedExperimentLogger(dataset_exp_path)
    max_i=False
    if 'goals' in dataset_exp_path:
        max_i=True
    dataset = dataset_logger.load_trans_dataset(max_i=max_i)
    dataset.set_pred_type('class')

    ds_pos_actions, ds_neg_actions, ds_labels, ds_heights = [], [], [], []
    for x,y in dataset:
        label = int(y.detach().numpy())
        ds_labels.append(label)
        str_action = ','.join([str(int(a.detach())) for a in x[2]])
        if label == 1: ds_pos_actions.append(str_action)
        if label == 0: ds_neg_actions.append(str_action)
        vef = x[1].detach().numpy()
        ds_heights.append(vef[1:,1:].sum()+1)  # one block is stacked on table

    for d, ds_list in zip([labels, pos_actions, neg_actions, heights], [ds_labels, ds_pos_actions, ds_neg_actions, ds_heights]):
        for key in d:
            d[key].append(ds_list.count(key))

# calc mean and std dev over all datasets
for d in [labels, pos_actions, neg_actions, heights]:
    for key in d:
        mean = np.mean(d[key])
        std = np.std(d[key])
        d[key] = [mean, std]

# Label Frequency
plt.figure()
plt.bar(labels.keys(), [ms[0] for ms in labels.values()], yerr=[ms[1] for ms in labels.values()])
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.ylim(0, 90)
plt.savefig('1')
plt.close()

# Transition Frequency
plt.figure()
for actions, label in zip([pos_actions, neg_actions], ['Positive', 'Negative']):
    plt.bar(list(actions.keys()),
            [ms[0] for ms in actions.values()],
            label=label,
            color='r' if label=='Negative' else 'g',
            yerr=[ms[1] for ms in actions.values()])
plt.xlabel('Action')
plt.ylabel('Frequency')
plt.ylim(0, 37)
plt.legend()
plt.savefig('2')
plt.close()

# Tower Heights
plt.figure()
plt.bar(heights.keys(), [ms[0] for ms in heights.values()], yerr=[ms[1] for ms in heights.values()])
plt.xlabel('Tower Heights')
plt.ylabel('Frequency')
plt.ylim(0, 100)
plt.savefig('3')
plt.close()
