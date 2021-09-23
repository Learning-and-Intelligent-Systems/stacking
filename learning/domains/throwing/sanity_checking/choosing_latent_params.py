import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from learning.domains.throwing.throwing_data import label_actions
from learning.domains.throwing.entities import ThrowingAction, ThrowingBall

data_save_folder = "learning/domains/throwing/sanity_checking/data/"
data_save_file = "choosing_latent_params_rotated_obstacle.npy"
data_save_path = os.path.join(data_save_folder, data_save_file)

attributes = [
    "mass",
    "radius",
    "air_drag_linear",
    "air_drag_angular",
    "friction_coef",
    "rolling_resistance",
    "bounciness",
]

ranges = [
    (0.5, 1.5),
    (0.02, 0.06),
    (0, 2),
    (5e-6, 5e-5),
    (0.1, 1),
    (1e-4, 1e-3),
    (0.1, 0.8),
]

n_attributes = len(attributes)
n_samples = 100
n_range = 100

# load or rebuild the data
try:
    data = np.load(data_save_path)

except FileNotFoundError:
    data = np.zeros([n_attributes, n_samples, n_range])
    for i in range(n_attributes):
        # choose some random actions
        for j, a in tqdm(enumerate(ThrowingAction.random_vector(n_samples=n_samples)), total=n_samples):
            # for each action, plot the latent value vs the distance traveled
            objects = [ThrowingBall(**{attributes[i]: param_value}) for param_value in np.linspace(*ranges[i], n_range)]
            actions = np.tile(a[None,:], [n_range, 1])
            z_ids = np.arange(n_range)
            data[i,j] = label_actions(objects, actions, z_ids)

    os.makedirs(data_save_folder, exist_ok=True)
    np.save(data_save_path, data)

fig, axes = plt.subplots(nrows=2, ncols=4)

for i in range(n_attributes):
    for j in range(n_samples):
        axes.flat[i].plot(np.linspace(*ranges[i], n_range), data[i, j], c='b', alpha=0.3)

    axes.flat[i].set_xlabel(attributes[i])
    axes.flat[i].set_ylabel("Distance (m)")

plt.suptitle("How distance is affected by hidden params (rotated obstacle)")
plt.show()