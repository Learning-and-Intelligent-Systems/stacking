import numpy as np
from matplotlib import pyplot as plt
import os

from learning.active.utils import ActiveExperimentLogger
from learning.domains.throwing.entities import ThrowingBall


exp_dir = "learning/experiments/logs/sept_27th"
good_runs = ['throwing_20_objects_repeat_run_3-20210927-101735',
             'throwing_20_objects_repeat_run_2-20210927-101735',
             'throwing_20_objects_repeat_run_1-20210927-101735']
bad_runs = ['throwing_20_objects_repeat_run_0-20210927-101736',
            'throwing_20_objects_repeat_run_4-20210927-101736']

save_dir = "learning/domains/throwing/sanity_checking/figures/probing_initial_dataset"
def save_or_show(name, save_dir=None):
    if save_dir is None:
        plt.show()
    else:
        plt.savefig(os.path.join(save_dir, name))
        plt.clf()


# plot the good and bad runs
for i, r in enumerate(good_runs + bad_runs):
    path_to_val_acc = os.path.join(exp_dir, r, 'results/val_accuracy.npy')
    plt.plot(np.load(path_to_val_acc), c='g' if i < len(good_runs) else 'r')
plt.title("Good and Bad runs validation accuracy")
save_or_show("val_acc.png",save_dir=save_dir)


# open experiment loggers for each log
good_loggers = [ActiveExperimentLogger(os.path.join(exp_dir, r)) for r in good_runs]
bad_loggers = [ActiveExperimentLogger(os.path.join(exp_dir, r)) for r in bad_runs]
for logger in good_loggers + bad_loggers:
    logger.args.throwing = True
    logger.args.max_acqusitions = 100


# plot the object distributions for each log
fig, axes = plt.subplots(nrows=2, ncols=max(len(good_runs), len(bad_runs)))
for i, set_of_loggers in enumerate([good_loggers, bad_loggers]):
    for j, logger in enumerate(set_of_loggers):
        objects = logger.get_objects(ThrowingBall)
        xs = [o.air_drag_linear for o in objects]
        ys = [o.rolling_resistance for o in objects]
        axes[i,j].scatter(xs, ys, c='g' if i==0 else 'r')
        axes[i,j].set_xlim(0,2)
        axes[i,j].set_ylim(1e-4, 1e-3)
plt.suptitle('Object Hidden Params')
save_or_show("object_distribution.png",save_dir=save_dir)


# plot the action distributions
fig, axes = plt.subplots(nrows=2, ncols=max(len(good_runs), len(bad_runs)))
for i, set_of_loggers in enumerate([good_loggers, bad_loggers]):
    for j, logger in enumerate(set_of_loggers):
        init_dataset = logger.load_dataset(0).tensors[0][:,-2:]
        # axes[i,j].cla()
        axes[i,j].scatter(*init_dataset.T, c='g' if i==0 else 'r')
        axes[i,j].set_xlim(np.pi/8, 3*np.pi/8)
        axes[i,j].set_ylim(-10, 10)
plt.suptitle('Action Parameters')
save_or_show("action_distribution.png",save_dir=save_dir)

for i, logger in enumerate(good_loggers + bad_loggers):
    ys = np.sort(logger.load_dataset(0).tensors[2].squeeze().numpy())
    plt.plot(ys, c='g' if i < len(good_runs) else 'r')
plt.title("All labels for each run, sorted.")
save_or_show("sorted_labels.png",save_dir=save_dir)