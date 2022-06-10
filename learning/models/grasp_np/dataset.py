import numpy as np
import torch

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


class MultiTargetGNPGraspDataset(Dataset):
    def __init__(self, data, context_data=None):
        """ Given data dictionaries, format them for the GNP model.

        If just data is given, generate a context for each grasp that includes all 
        the other contexts for that object.

        If context_data is also given, use all of context_data as the contexts and 
        only data as the targets. Note that object IDs should correspond in each 
        dataset for proper functionality.
        """
        
        # Each of these is a list of length #objects.
        self.meshes, self.context_pool_with_labels, self.heldout_pool_with_labels = self.process_raw_data(data, context_data)

    def process_raw_data(self, data, context_data):
        meshes = self.get_meshes_from_dataset(data)
        if context_data is None:
            context_pool_with_labels = self.get_grasps_from_dataset(data)
            heldout_pool_with_labels = [None]*len(meshes)
        else:
            context_pool_with_labels = self.get_grasps_from_dataset(context_data)
            heldout_pool_with_labels = self.get_grasps_from_dataset(data)
        return meshes, context_pool_with_labels, heldout_pool_with_labels

    def get_meshes_from_dataset(self, data):
        object_data = data['object_data']
        all_object_ixs = np.array(data['grasp_data']['object_ids'])
        all_Xs = np.array(data['grasp_data']['grasps']).astype('float32')

        all_meshes = []
        for ox in range(len(object_data['object_names'])):
            object_ixs = (all_object_ixs == ox)
            Xs = all_Xs[object_ixs, ...]    
            meshes_ox = Xs[:, 3:, :3]
            meshes_ox = np.concatenate([
                meshes_ox,
                np.zeros((meshes_ox.shape[0], meshes_ox.shape[1], 2))
            ], axis=2).astype('float32')
            all_meshes.append(meshes_ox)
        return all_meshes

    def get_grasps_from_dataset(self, data):
        object_data = data['object_data']
        all_object_ixs = np.array(data['grasp_data']['object_ids'])
        all_Xs = np.array(data['grasp_data']['grasps']).astype('float32')
        all_labels = np.array(data['grasp_data']['labels']).astype('float32')

        all_grasps = []
        for ox in range(len(object_data['object_names'])):
            object_ixs = (all_object_ixs == ox)
            Xs = all_Xs[object_ixs, ...]    
            grasps_ox = (Xs[:, 0, :3] + Xs[:, 1, :3])/2.
            labels_ox = all_labels[object_ixs]

            grasps_ox = np.concatenate([
                grasps_ox,
                np.ones((grasps_ox.shape[0], 1)),
                labels_ox[:, None]
            ], axis=1).astype('float32')

            all_grasps.append(grasps_ox)
        return all_grasps

    def __getitem__(self, ox):
        return self.meshes[ox], self.context_pool_with_labels[ox], self.heldout_pool_with_labels[ox]

    def __len__(self):
        return len(self.meshes)


def collate_fn(items):
    """
    Decide how many context and target points to add.
    """
    if items[0][2] is not None:
        n_context = items[0][1].shape[0]
        n_target = items[0][2].shape[0]
    else:
        max_context = items[0][1].shape[0] + 1
        n_context = np.random.randint(max_context)
        max_target = max_context - n_context
        n_target = np.random.randint(max_target)
    print(f'n_context: {n_context}\tn_target: {n_target}')

    contexts, target_xs, target_ys = [], [], []
    for meshes, context_pool, heldout_pool in items:
        if heldout_pool is None:
            # We are training and will reuse context pool.
            random_ixs = np.random.permutation(context_pool.shape[0]) 
            context_ixs = random_ixs[:n_context]
            target_ixs = random_ixs[n_context:(n_context+n_target)]

            mesh_ix = np.random.randint(0, meshes.shape[0])
            mesh = meshes[mesh_ix, ...]

            context = np.concatenate([
                context_pool[context_ixs],
                mesh
            ], axis=0).astype('float32')
            contexts.append(context.T)

            # Create target_xs: remove labels from all grasps.
            target_x = np.concatenate([
                context_pool[target_ixs],
                context_pool[context_ixs],
                mesh
            ], axis=0).astype('float32')[:, :-1]
            target_xs.append(target_x.T)

            target_y = np.concatenate([
                context_pool[target_ixs, 4],
                context_pool[context_ixs, 4],
            ], axis=0).astype('float32')
            target_ys.append(target_y)
        
        else:
            # We are testing and will keep context and targets separate.
            mesh_ix = np.random.randint(0, meshes.shape[0])
            mesh = meshes[mesh_ix, ...]

            context = np.concatenate([
                context_pool,
                mesh
            ], axis=0).astype('float32')
            contexts.append(context.T)

            # Create target_xs: remove labels from all grasps.
            target_x = np.concatenate([
                heldout_pool,
                mesh
            ], axis=0).astype('float32')[:, :-1]
            target_xs.append(target_x.T)

            target_y = heldout_pool[:, 4]
            target_ys.append(target_y)

    return torch.Tensor(contexts), torch.Tensor(target_xs), torch.Tensor(target_ys)


class CustomGNPGraspDataset(Dataset):
    def __init__(self, data, context_data=None):
        """ Given data dictionaries, format them for the GNP model.

        If just data is given, generate a context for each grasp that includes all 
        the other contexts for that object.

        If context_data is also given, use all of context_data as the contexts and 
        only data as the targets. Note that object IDs should correspond in each 
        dataset for proper functionality.
        """
        
        # Each of these is a list of length #objects.
        self.cp_grasp_geometries, self.cp_grasp_midpoints, self.cp_grasp_labels = self.process_raw_data(context_data)
        self.hp_grasp_geometries, self.hp_grasp_midpoints, self.hp_grasp_labels = self.process_raw_data(data)

    def process_raw_data(self, data):
        if data is None:
            return None, None, None
        else:            
            grasp_geometries = {k: np.array(v).astype('float32') for k, v in data['grasp_data']['grasp_geometries'].items()}
            grasp_midpoints = {k: np.array(v).astype('float32') for k, v in data['grasp_data']['grasp_midpoints'].items()}
            grasp_labels = {k: np.array(v).astype('float32') for k, v in data['grasp_data']['labels'].items()}
            return grasp_geometries, grasp_midpoints, grasp_labels

    def __getitem__(self, ox):
        if self.cp_grasp_geometries is None:
            cp_data = None
        else:
            cp_data = {
                'grasp_geometries': self.cp_grasp_geometries[ox],
                'grasp_midpoints': self.cp_grasp_midpoints[ox],
                'grasp_labels': self.cp_grasp_labels[ox]
            }
        hp_data = {
            'grasp_geometries': self.hp_grasp_geometries[ox],
            'grasp_midpoints': self.hp_grasp_midpoints[ox],
            'grasp_labels': self.hp_grasp_labels[ox]
        }
        return cp_data, hp_data

    def __len__(self):
        return len(self.hp_grasp_geometries)


def custom_collate_fn(items):
    """
    Decide how many context and target points to add.
    """
    if items[0][0] is not None:
        n_context = items[0][0]['grasp_geometries'].shape[0]
        n_target = items[0][1]['grasp_geometris'].shape[0]
    else:
        max_context = items[0][1]['grasp_geometries'].shape[0] + 1
        n_context = np.random.randint(max_context)
        max_target = max_context - n_context
        n_target = np.random.randint(max_target)
    print(f'n_context: {n_context}\tn_target: {n_target}')

    contexts, target_xs, target_ys = [], [], []
    for meshes, context_pool, heldout_pool in items:
        if heldout_pool is None:
            # We are training and will reuse context pool.
            random_ixs = np.random.permutation(context_pool.shape[0]) 
            context_ixs = random_ixs[:n_context]
            target_ixs = random_ixs[n_context:(n_context+n_target)]

            mesh_ix = np.random.randint(0, meshes.shape[0])
            mesh = meshes[mesh_ix, ...]

            context = np.concatenate([
                context_pool[context_ixs],
                mesh
            ], axis=0).astype('float32')
            contexts.append(context.T)

            # Create target_xs: remove labels from all grasps.
            target_x = np.concatenate([
                context_pool[target_ixs],
                context_pool[context_ixs],
                mesh
            ], axis=0).astype('float32')[:, :-1]
            target_xs.append(target_x.T)

            target_y = np.concatenate([
                context_pool[target_ixs, 4],
                context_pool[context_ixs, 4],
            ], axis=0).astype('float32')
            target_ys.append(target_y)
        
        else:
            # We are testing and will keep context and targets separate.
            mesh_ix = np.random.randint(0, meshes.shape[0])
            mesh = meshes[mesh_ix, ...]

            context = np.concatenate([
                context_pool,
                mesh
            ], axis=0).astype('float32')
            contexts.append(context.T)

            # Create target_xs: remove labels from all grasps.
            target_x = np.concatenate([
                heldout_pool,
                mesh
            ], axis=0).astype('float32')[:, :-1]
            target_xs.append(target_x.T)

            target_y = heldout_pool[:, 4]
            target_ys.append(target_y)

    return torch.Tensor(contexts), torch.Tensor(target_xs), torch.Tensor(target_ys)


if __name__ == '__main__':
    import pickle
    from torch.utils.data import DataLoader

    train_dataset_fname = 'learning/data/grasping/train-sn100-test-sn10-robust/grasps/training_phase/train_grasps.pkl'
    val_dataset_fname = 'learning/data/grasping/train-sn100-test-sn10-robust/grasps/training_phase/val_grasps.pkl'
    print('Loading train dataset...')
    with open(train_dataset_fname, 'rb') as handle:
        train_data = pickle.load(handle)
    print('Loading val dataset...')
    with open(val_dataset_fname, 'rb') as handle:
        val_data = pickle.load(handle)
    
    train_dataset = MultiTargetGNPGraspDataset(data=train_data)
    val_dataset = MultiTargetGNPGraspDataset(data=val_data, context_data=train_data)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=128,
        collate_fn=collate_fn,
        shuffle=True
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        collate_fn=collate_fn,
        batch_size=64,
        shuffle=False
    )

    for batch in train_dataloader:
        print(len(batch))
        for elem in batch:
            print(elem.shape)

    for batch in val_dataloader:
        print(len(batch))
        for elem in batch:
            print(elem.shape)