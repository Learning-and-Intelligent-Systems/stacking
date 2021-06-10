import numpy as np
import torch

from agents.throwing_agent import ThrowingAgent
from learning.domains.throwing.entities import ThrowingBall, ThrowingAction

def sample_action(obj_ids, n_samples=1):
    z_ids = np.random.choice(a=obj_ids, size=n_samples, replace=True)
    ang = np.random.uniform(np.pi/8, 3*np.pi/8, size=n_samples)
    w = np.random.uniform(-10, 10, size=n_samples)
    return np.stack([ang, w], axis=1), z_ids

def label_actions(objects, xs, z_ids):
    agent = ThrowingAgent(objects)
    y = []

    for x, z_id in zip(xs, z_ids):
        b = objects[z_id]
        act = ThrowingAction.from_vector(b, x)
        y.append(agent.run(act))

    return np.array(y)

def generate_dataset(objects, n_data, as_tensor=True):
    obj_ids = np.arange(len(objects))
    xs, z_ids = sample_action(obj_ids, n_samples=n_data)
    ys = label_actions(objects, xs, z_ids)
    dataset = xs, z_ids, ys
    return (torch.Tensor(d) for d in dataset) if as_tensor else dataset

def generate_objects(n_objects):
    return [ThrowingBall.random() for _ in range(n_objects)]