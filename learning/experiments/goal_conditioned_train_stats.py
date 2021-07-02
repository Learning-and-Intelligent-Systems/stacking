import argparse
from argparse import Namespace
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import numpy as np

from learning.experiments.goal_conditioned_train import generate_world_dataset, \
                                                        setup_and_train, evaluate
        
args = Namespace(
            debug=False,
            domain='abc_blocks',
            goals_gile_path='learning/domains/abc_blocks/goal_files/goals_05052021.csv',
            max_seq_attempts=10,
            pred_type='delta_state',
            exp_name='test',
            plot=False,
            train_num_blocks=None,
            test_num_blocks=None)

num_models = 5 # must be odd and > 3
all_train_num_blocks = [2, 3, 4, 5]
all_test_num_blocks = [2, 3, 4, 5, 6, 7, 8]

train_datasets = {}
for i in range(num_models):
    for train_num_blocks in all_train_num_blocks:
        print('Generating training dataset with %i blocks' % train_num_blocks)
        train_datasets[(i, train_num_blocks)] = generate_world_dataset(args, train_num_blocks)
    
test_datasets = {}
for test_num_blocks in all_test_num_blocks:
    print('Generating test dataset with %i blocks' % train_num_blocks)
    test_datasets[test_num_blocks] = generate_world_dataset(args, test_num_blocks)
    
trans_models = {}
for (i, train_num_blocks), (train_dataset, _) in train_datasets.items():
    trans_models[(i, train_num_blocks)] = setup_and_train(args, train_dataset)
    
# Evaluate
print('Evaluating')
all_perc_explored = {}
all_accuracies = {}
for (i, train_num_blocks), (train_dataset, train_world) in train_datasets.items():
    for test_num_blocks, (test_dataset, test_world) in test_datasets.items():
        perc_t_explored, test_accuracy = evaluate(args, 
                                                trans_models[(i, train_num_blocks)], 
                                                train_dataset, 
                                                test_dataset, train_world)
        all_perc_explored[(train_num_blocks, test_num_blocks, i)] = perc_t_explored
        all_accuracies[(train_num_blocks, test_num_blocks, i)] = test_accuracy
            
# Plot
figure, ax = plt.subplots()
cs = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
for ci, train_num_blocks in enumerate(all_train_num_blocks):
    min_accuracies = []
    mid_accuracies = []
    max_accuracies = []
    for test_num_blocks in all_test_num_blocks:
        all_model_accs = [all_accuracies[(train_num_blocks, test_num_blocks, i)] for i in range(num_models)]
        print(all_model_accs)
        sorted_model_accs = np.sort(all_model_accs)
        m = round(num_models/2)
        min_accuracies.append(sorted_model_accs[0])
        max_accuracies.append(sorted_model_accs[-1])
        mid_accuracies.append(sorted_model_accs[m]) 
    ax.plot(all_test_num_blocks, mid_accuracies, color=cs[ci], label=str(train_num_blocks))
    ax.fill_between(all_test_num_blocks, min_accuracies, max_accuracies, color=cs[ci], alpha=0.1)
ax.set_xlabel('Number of Test Blocks')
ax.set_ylabel('Accuracy')
ax.legend(title='Number of Training Blocks')

plt.show()
