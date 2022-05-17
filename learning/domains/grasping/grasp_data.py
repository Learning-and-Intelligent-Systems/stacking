import numpy as np

from torch.utils.data import Dataset, DataLoader, Sampler

from learning.domains.grasping.generate_grasp_datasets import graspablebody_from_vector
from pb_robot.planners.antipodalGraspPlanner import GraspSimulationClient


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

        if grasp_encoding == 'per_point':
            self.grasp_vectors = self.get_per_point_repr(self.grasp_vectors)

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

        
def visualize_grasp_dataset(dataset, ycb_name, n_objects, n_grasps_per_object, labels=None, figure_path=''):
    if labels is None:
        labels = dataset.grasp_labels

    for ix in range(n_objects):

        object_properties = dataset.grasp_vectors[ix*n_grasps_per_object][0, -5:]
        graspable_body = graspablebody_from_vector(ycb_name, object_properties)

        sim_client = GraspSimulationClient(graspable_body, False, 'object_models')
        grasps = []
        for gx in range(n_grasps_per_object):
            grasp_points = []
            for kx in range(3):
                grasp_points.append(dataset.grasp_vectors[ix*n_grasps_per_object+gx][kx, 0:3])
            grasps.append(grasp_points)

        obj_labels = labels[ix*n_grasps_per_object:(ix+1)*n_grasps_per_object]

        if len(figure_path) > 0:
            sim_client.tm_show_grasps(grasps, obj_labels, fname=figure_path % ix)
        else:
            sim_client.tm_show_grasps(grasps, obj_labels)

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
