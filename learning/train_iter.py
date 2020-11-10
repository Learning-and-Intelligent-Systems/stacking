from argparse import Namespace
import pickle
import matplotlib.pyplot as plt
import numpy as np

from learning.train_gat import main

num_exps = 3
args = Namespace(visual=True, batch_size=64, epochs=50)
all_train_losses = []
all_test_accs = []

for _ in range(num_exps):
    train_losses, epoch_ids, test_accuracies, test_num_blocks = main(args)
    all_train_losses += [train_losses]
    all_test_accs += [test_accuracies]

with open('results.pickle', 'wb') as handle:
    pickle.dump([all_train_losses, all_test_accs, epoch_ids], handle)
    
# plot training
fig_train, ax_train = plt.subplots()
for i in range(num_exps):
    ax_train.plot(list(range(len(all_train_losses[i]))), all_train_losses[i])
    ax_train.set_xlabel('Batch #')
    ax_train.set_ylabel('Train Losses')
fig_train.savefig('train_losses.png')

# plot testing
for block_ind, num_blocks in enumerate(test_num_blocks):
    fig_test, ax_test = plt.subplots()
    for exp in range(num_exps):
        block_accs = np.array(all_test_accs[exp]).T[block_ind]
        ax_test.plot(epoch_ids, block_accs)
    ax_test.set_title(str(num_blocks)+' blocks')
    ax_test.set_xlabel('Epoch #')
    ax_test.set_ylabel('Test Accuracies')
    
    fig_test.savefig('test_accuracies_'+str(num_blocks)+'blocks.png')
