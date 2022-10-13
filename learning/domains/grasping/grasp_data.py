import numpy as np
import pickle

from torch.utils.data import Dataset, DataLoader, Sampler

from learning.domains.grasping.generate_grasp_datasets import graspablebody_from_vector
from pb_robot.planners.antipodalGraspPlanner import GraspSimulationClient

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class GraspDataset(Dataset):
    def __init__(self, data, grasp_encoding='as_points'):
        """
        :grasp_encoding: 'as_point' encodes the grasp as the first three points (as is).
            'per_point' includes the grasp in the feature vector of each point.
        """
        self.data = data

        self.grasp_vectors = np.array(data['grasp_data']['grasps']).astype('float32')
        self.grasp_object_ids = data['grasp_data']['object_ids']
        self.grasp_labels = np.array(data['grasp_data']['labels']).astype('float32')
        print(self.grasp_vectors.shape)
        if grasp_encoding == 'per_point':
            self.grasp_vectors = self.get_per_point_repr(self.grasp_vectors)
            #self.grasp_vectors = self.remove_far_points(self.grasp_vectors, 0.02)
        
    def remove_far_points(self, grasp_vectors, threshold):
        new_grasp_vectors = []

        for gx in range(grasp_vectors.shape[0]):    
            new_grasp = []
            grasp = grasp_vectors[gx]
            # finger1 = grasp[0, 6:9]
            # finger2 = grasp[0, 9:12]
            finger1 = grasp[0, 0:3]
            finger2 = grasp[1, 0:3]
            for px in range(3, grasp.shape[0]):
                xyz = grasp[px][0:3]
                d1 = np.linalg.norm(finger1-xyz)
                d2 = np.linalg.norm(finger2-xyz)
                if d1 < threshold or d2 < threshold:
                    new_grasp.append(grasp[px])
            new_grasp_vectors.append(new_grasp)
            if gx % 50 == 0:
                fig = plt.figure()
                ax = plt.axes(projection='3d')
                lim = 0.2
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
                ax.set_zlim(-lim, lim)
                points = np.array(new_grasp)
                ax.scatter(grasp[3:, 0], grasp[3:, 1], grasp[3:, 2], color='b', alpha=0.2)
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='r', alpha=1.0)
                plt.show()
            print(f'Reduced grasps from {512} to {len(new_grasp)}')
        new_grasps_vectors = np.array(new_grasp_vectors, dtype='float32')
        
        return new_grasps_vectors

    def get_per_point_repr(self, grasp_vectors):
        """ By default grasps_vectors is of shape (N_grasps, (3+N_points), 3+N_feats).
        """
        new_repr = []
        for gx in range(grasp_vectors.shape[0]):
            grasp_vector = grasp_vectors[gx, :, :]
            grasp_points = grasp_vector[0:3, 0:3].flatten()

            object_points = grasp_vector[3:, 0:3]
            object_properties = grasp_vector[3:, 6:]
            
            N_points = object_points.shape[0]
            grasp_points = np.tile(grasp_points[None, :], (N_points, 1))
        
            new_grasp_vector = np.concatenate([object_points, grasp_points, object_properties], axis=1)
            new_repr.append(new_grasp_vector)

        return np.array(new_repr)

        
    def __getitem__(self, ix):
        """
        Return a DxN tensor.
        """
        return (self.grasp_vectors[ix].T, self.grasp_object_ids[ix], self.grasp_labels[ix])

    def __len__(self):
        return len(self.grasp_vectors)

    def add_to_dataset(self):
        pass


class GraspParallelDataLoader:
    def __init__(self, dataset, batch_size, shuffle, n_dataloaders=1):
        """ Wrap multiple dataloaders so that we iterate through the data independently and in parallel.
        :param dataset: The underlying dataset to iterate over.
        :param batch_size: Batch size.
        :param shuffle: Whether to shuffle the data.
        :param n_dataloaders: The number of underlying independent dataloaders to use.
        """
        # Create a custom sampler and loader so each loader uses idependently shuffled data.
        self.loaders = []
        for _ in range(n_dataloaders):
            loader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=shuffle)
            self.loaders.append(loader)

    def __iter__(self):
        # Initialize the dataloaders (this should reshuffle the data in each).
        loaders = [iter(l) for l in self.loaders]
        stop = False
        # Return a separate batch for each loader at each step.
        while not stop:
            batches = []
            for loader in loaders:
                try:
                    batches.append(next(loader))
                except:
                    # print('[ParallelDataLoader] Warning: failed to get batch from all loaders.')
                    stop = True
            if not stop:
                yield batches

    def __len__(self):
        return len(self.loaders[0])


def visualize_acquisition_dataset(logger, figure_path=''):

    grasps, obj_labels = [], []
    for tx in range(1, 11):
        dataset, _ = logger.load_acquisition_data(tx)
        print(type(dataset))

        graspable_body = dataset['grasp_data']['raw_grasps'][0].graspable_body
        
        grasp_points = []
        for kx in range(3):
            grasp_points.append(dataset['grasp_data']['grasps'][0][kx, 0:3])
        grasps.append(grasp_points)

        obj_labels.append(dataset['grasp_data']['labels'][0])

    print(obj_labels)
    sim_client = GraspSimulationClient(graspable_body, False, 'object_models')
    if len(figure_path) > 0:
        sim_client.tm_show_grasps(grasps, obj_labels, fname=figure_path % ix)
    else:
        sim_client.tm_show_grasps(grasps, obj_labels)
    sim_client.disconnect()

if __name__ == '__main__':
    logger_fname = 'learning/experiments/logs/grasp_train-sn-test-ycb-1_fit_bald_train_geo_object11-20220509-233743'
    from learning.active.utils import ActiveExperimentLogger
    logger = ActiveExperimentLogger(logger_fname, use_latents=True)
    visualize_acquisition_dataset(logger)
