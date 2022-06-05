import numpy as np

from torch.utils.data import Dataset, DataLoader

class GNPGraspDataset(Dataset):
    def __init__(self, data):
        self.contexts, self.target_xs, self.target_ys = self.process_raw_data(data)

    def process_raw_data(self, data):
        # Group all grasps for each object first.
        grasp_data, object_data = data['grasp_data'], data['object_data']
        
        all_grasps = np.array(grasp_data['grasps']).astype('float32')
        all_object_ixs = np.array(grasp_data['object_ids'])
        all_labels = np.array(grasp_data['labels']).astype('float32')

        contexts = []
        target_xs = self.get_per_point_repr(all_grasps)
        target_ys = all_labels

        for ox in range(len(object_data['object_names'])):
            grasp_ixs = (all_object_ixs == ox)
            
            X_ox = all_grasps[grasp_ixs, ...]
            labels_ox = all_grasps[grasp_ixs]
            
            meshes_ox = X_ox[:, 3:, :3]
            grasps_ox = X_ox[:, 0:3, :3]
            midpoints_ox = (X_ox[:, 0, :] + X_ox[:, 1, :])/2.

            for gx in range(grasps_ox.shape[0]):
                # TODO: Create context vector leaving out grasp gx.
                context_meshpoints = meshes_ox[gx, ...]

                context_ixs = (np.arange(len(midpoints_ox)) != gx)
                context_grasppoints = midpoints_ox[context_ixs, ...]
                context_labels = labels_ox[context_ixs]
                
                # Add context indicators so we can differentiate the mesh from grasp points.
                context_mesh = np.concatenate([
                    context_meshpoints, 
                    np.zeros((context_meshpoints.shape[0], 1)),
                    np.zeros((context_meshpoints.shape[0], 1))  
                ], axis=1).astype('float32')
                context_grasps = np.concatenate([
                    context_grasppoints,
                    np.ones((context_grasppoints.shape[0], 1)),
                    context_labels[:, None]
                ], axis=1).astype('float32')
                context = np.concatenate([
                    context_mesh,
                    context_grasps
                ], axis=0)
                contexts.append(context)

        return contexts, target_xs, target_ys

    def get_per_point_repr(self, grasp_vectors):
        """ By default grasps_vectors is of shape (N_grasps, (3+N_points), 3+N_feats).
        """
        new_repr = []
        for gx in range(grasp_vectors.shape[0]):
            grasp_vector = grasp_vectors[gx, :, :]
            grasp_points = grasp_vector[0:3, 0:3]
            grasp_midpoint = (grasp_points[0, :] + grasp_points[1, :])/2. 

            object_points = grasp_vector[3:, 0:3]
            object_properties = grasp_vector[3:, 6:]
            
            N_points = object_points.shape[0]
            grasp_midpoint = np.tile(grasp_midpoint[None, :], (N_points, 1))
        
            new_grasp_vector = np.concatenate([object_points, grasp_midpoint, object_properties], axis=1)
            new_repr.append(new_grasp_vector)

        return np.array(new_repr).astype('float32')

    def __getitem__(self, ix):
        return self.contexts[ix], self.target_xs[ix], self.target_ys[ix]


