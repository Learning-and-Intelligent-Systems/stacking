import numpy as np

from torch.utils.data import Dataset, DataLoader

class GNPGraspDataset(Dataset):
    def __init__(self, data, context_data=None):
        """ Given data dictionaries, format them for the GNP model.

        If just data is given, generate a context for each grasp that includes all 
        the other contexts for that object.

        If context_data is also given, use all of context_data as the contexts and 
        only data as the targets. Note that object IDs should correspond in each 
        dataset for proper functionality.
        """
        
        self.contexts, self.target_xs, self.target_ys = self.process_raw_data(data, context_data)

    def process_raw_data(self, data, context_data=None):        
        # Targets always come from data.
        target_xs = self.get_per_point_repr(np.array(data['grasp_data']['grasps']).astype('float32'))
        target_ys = np.array(data['grasp_data']['labels']).astype('float32')

        object_data = data['object_data']
        # Choose the contexts from the appropriate dataset.
        if context_data is None:
            target_grasp_data = data['grasp_data']
            context_grasp_data = data['grasp_data']
        else:
            target_grasp_data = data['grasp_data']
            context_grasp_data = context_data['grasp_data']

        all_context_grasps = np.array(context_grasp_data['grasps']).astype('float32')
        all_context_object_ixs = np.array(context_grasp_data['object_ids'])
        all_context_labels = np.array(context_grasp_data['labels']).astype('float32')

        all_target_grasps = np.array(target_grasp_data['grasps']).astype('float32')
        all_target_object_ixs = np.array(target_grasp_data['object_ids'])

        contexts = []
        for ox in range(len(object_data['object_names'])):
            context_ixs = (all_context_object_ixs == ox)
            target_ixs = (all_target_object_ixs == ox)

            target_X_ox = all_target_grasps[target_ixs, ...]
            context_X_ox = all_context_grasps[context_ixs, ...]
            context_labels_ox = all_context_labels[context_ixs]
            
            # Meshes will come from the target set.
            meshes_ox = target_X_ox[:, 3:, :3]

            # Get grasps from the context grasp set.
            grasps_ox = context_X_ox[:, 0:3, :3]
            midpoints_ox = (grasps_ox[:, 0, :] + grasps_ox[:, 1, :])/2.

            for gx in range(meshes_ox.shape[0]):
                # TODO: Create context vector leaving out grasp gx.
                context_meshpoints = meshes_ox[gx, ...]

                if context_data is None:
                    # Exclude the grasp from contexts if we're using the same data for contexts.
                    context_ixs = (np.arange(len(midpoints_ox)) != gx)
                    context_grasppoints = midpoints_ox[context_ixs, ...]
                    context_labels = context_labels_ox[context_ixs]
                else:
                    context_grasppoints = midpoints_ox
                    context_labels = context_labels_ox
                
                # import IPython
                # IPython.embed()
                # sys.exit()
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
                    context_grasps,
                    context_mesh
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
        return self.contexts[ix].T, self.target_xs[ix].T, self.target_ys[ix]

    def __len__(self):
        return len(self.contexts)


