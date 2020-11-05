from argparse import Namespace
import pickle
import matplotlib.pyplot as plt

from learning.train_gat import main

args = Namespace(visual=True, batch_size=32, epochs=50)
all_train_losses = []
all_test_accs = []

for _ in range(5):
    train_losses, epoch_ids, test_accuracies = main(args)
    all_train_losses += [train_losses]
    all_test_accs += [test_accuracies]

with open('results.pickle', 'wb') as handle:
    pickle.dump([all_train_losses, all_test_accs, epoch_ids], handle)
    
# plot
fig_train, ax_train = plt.subplots()
fig_test, ax_test = plt.subplots()
for i in range(5):
    ax_train.plot(list(range(len(all_train_losses[i]))), all_train_losses[i])
    ax_train.set_xlabel('Batch #')
    ax_train.set_ylabel('Train Losses')
    
    ax_test.plot(list(range(len(all_test_accs[i]))), all_test_accs[i])
    ax_test.set_xlabel('Epoch #')
    ax_test.set_ylabel('Test Accuracies')
    
fig_train.savefig('train_losses.png')
fig_test.savefig('test_accuracies.png')
