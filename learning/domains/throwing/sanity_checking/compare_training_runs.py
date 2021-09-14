from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import os
from multiprocessing import Process
import torch

from learning.domains.throwing.train_latent import main, get_parser

#create a default args object that we'll modify for each run
args = get_parser().parse_args()
args.n_train = 500
args.n_val = 100
args.n_epochs = 200
args.n_objects = 10
args.use_normalization = True

labels = ['fully_observed', 'restitution_hidden', 'radius_hidden', 'restitution_and_radius_hidden']
hide_dims = ['', '9', '4', '9,4']

processes = []

n_max = 4
d_max = 4

# execute the commands in parallel using multiprocess
for n in range(n_max):
    for l, h in zip(labels, hide_dims):
        for d in range(d_max):
            args.d_latent = d
            args.hide_dims = h
            args.save_accs = f"learning/domains/throwing/sanity_checking/data/500_epochs/{l}_{d}d_latent_run_{n}.npy"

            # main(args)
            if len(processes) > 13:
                print('[WARNING] Waiting for a process to complete!')
                processes[0].join()
                processes.pop(0)

            processes.append(Process(target=main, args=(deepcopy(args),)))
            np.random.seed() # this is critical!
            torch.random.seed() # this is critical!
            processes[-1].start()


for p in processes:
    p.join()



fig, axes = plt.subplots(nrows=2, ncols=2)

for l, ax in zip(labels, axes.flat):
    for d in range(d_max):
        # load the runs
        data_to_average = []
        for n in range(n_max):
            fname = f"learning/domains/throwing/sanity_checking/data/500_epochs/{l}_{d}d_latent_run_{n}.npy"
            data_to_average.append(np.load(fname))
        data_to_plot = np.array(data_to_average).mean(axis=0)
        ax.plot(data_to_plot, label=f'{d}D Latent')

    ax.legend()
    ax.set_title(l)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Likelihood')

plt.show()

# processes = []
# for n in range(10):
#     args = get_parser().parse_args()
#     args.n_train = 500
#     args.n_val = 100
#     args.n_epochs = 500
#     args.save_accs = f"learning/domains/throwing/sanity_checking/data/repeatability_test/run_{n}.npy"
#     processes.append(Process(target=main, args=(args,)))
#     np.random.seed() # this is critical!
#     torch.random.seed() # this is critical!
#     processes[-1].start()


# for p in processes:
#     p.join()


# for n in range(10):
#     fname = f"learning/domains/throwing/sanity_checking/data/repeatability_test/run_{n}.npy"
#     plt.plot(np.load(fname), c='b', alpha=0.3)

# plt.xlabel('Epoch')
# plt.ylabel('Log-Likelihood')
# plt.title('Consistency of Multiple Training Runs')
# plt.show()
