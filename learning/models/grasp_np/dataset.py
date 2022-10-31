import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader


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
        self.cp_grasp_geometries, self.cp_grasp_midpoints, self.cp_grasp_labels, self.cp_full_meshes = self.process_raw_data(context_data)
        self.hp_grasp_geometries, self.hp_grasp_midpoints, self.hp_grasp_labels, self.hp_full_meshes = self.process_raw_data(data)
        self.object_indices = sorted(self.hp_grasp_geometries.keys())

    def process_raw_data(self, data):
        if data is None:
            return None, None, None, None
        else:            
            grasp_geometries = {}
            for k, v in data['grasp_data']['grasp_geometries'].items():
                meshes = [arr[:256, :] for arr in v]
                for mx in range(len(meshes)):
                    while meshes[mx].shape[0] != 256: 
                        n_dup = 256 - meshes[mx].shape[0]
                        meshes[mx] = np.concatenate([meshes[mx], meshes[mx][:n_dup, :]], axis=0)
                        
                grasp_geometries[k] = np.array(meshes).astype('float32')
            grasp_midpoints = {k: np.array(v).astype('float32') for k, v in data['grasp_data']['grasp_midpoints'].items()}
            grasp_labels = {k: np.array(v).astype('float32') for k, v in data['grasp_data']['labels'].items()}
            full_meshes = {k: np.array(v).astype('float32') for k, v in data['grasp_data']['object_meshes'].items() }
            return grasp_geometries, grasp_midpoints, grasp_labels, full_meshes

    def __getitem__(self, ix):
        ox = self.object_indices[ix]
        if self.cp_grasp_geometries is None:
            cp_data = None
        else:
            cp_data = {
                'object_mesh': self.cp_full_meshes[ox]/0.02,
                'grasp_geometries': self.cp_grasp_geometries[ox]/0.02,
                'grasp_midpoints': self.cp_grasp_midpoints[ox]/0.02,
                'grasp_labels': self.cp_grasp_labels[ox]
            }
        hp_data = {
            'object_mesh': self.hp_full_meshes[ox]/0.02,
            'grasp_geometries': self.hp_grasp_geometries[ox]/0.02,
            'grasp_midpoints': self.hp_grasp_midpoints[ox]/0.02,
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
        n_target = items[0][1]['grasp_geometries'].shape[0]
    else:
        max_context = items[0][1]['grasp_geometries'].shape[0] + 1
        n_context = np.random.randint(low=40, high=max_context)
        max_target = max_context - n_context
        n_target = np.random.randint(max_target)
    # print(f'n_context: {n_context}\tn_target: {n_target}')

    context_geoms, context_midpoints, context_labels = [], [], []
    target_geoms, target_midpoints, target_labels = [], [], []
    full_meshes = []


    for context_data, heldout_data in items:
        full_meshes.append(heldout_data['object_mesh'][0, :, :].swapaxes(0, 1))
        if context_data is None:
            all_context_geoms = heldout_data['grasp_geometries']
            all_context_midpoints = heldout_data['grasp_midpoints']
            all_context_labels = heldout_data['grasp_labels']

            # We are training and will reuse context pool.
            random_ixs = np.random.permutation(all_context_geoms.shape[0]) 
            context_ixs = random_ixs[:n_context]
            target_ixs = random_ixs[:(n_context+n_target)]

            context_geoms.append(all_context_geoms[context_ixs, ...].swapaxes(1, 2))
            context_midpoints.append(all_context_midpoints[context_ixs,...])
            context_labels.append(all_context_labels[context_ixs])

            target_geoms.append(all_context_geoms[target_ixs, ...].swapaxes(1, 2))
            target_midpoints.append(all_context_midpoints[target_ixs,...])
            target_labels.append(all_context_labels[target_ixs])
        else:
            # We are testing and will keep context and targets separate.
            context_geoms.append(context_data['grasp_geometries'].swapaxes(1, 2))
            context_midpoints.append(context_data['grasp_midpoints'])
            context_labels.append(context_data['grasp_labels'])

            target_geoms.append(heldout_data['grasp_geometries'].swapaxes(1, 2))
            target_midpoints.append(heldout_data['grasp_midpoints'])
            target_labels.append(heldout_data['grasp_labels'])
    
    context_geoms = np.array(context_geoms).astype('float32')
    context_midpoints = np.array(context_midpoints).astype('float32')
    context_labels = np.array(context_labels).astype('float32')

    target_geoms = np.array(target_geoms).astype('float32')
    target_midpoints = np.array(target_midpoints).astype('float32')
    target_labels = np.array(target_labels).astype('float32')

    full_meshes = np.array(full_meshes).astype('float32')
    return (torch.Tensor(context_geoms), torch.Tensor(context_midpoints), torch.Tensor(context_labels)),  (torch.Tensor(target_geoms), torch.Tensor(target_midpoints), torch.Tensor(target_labels)), torch.Tensor(full_meshes)


if __name__ == '__main__':
    import pickle
    from torch.utils.data import DataLoader

    train_dataset_fname = 'learning/data/grasping/train-sn100-test-sn10-robust-gnp/grasps/training_phase/train_grasps.pkl'
    val_dataset_fname = 'learning/data/grasping/train-sn100-test-sn10-robust-gnp/grasps/training_phase/val_grasps.pkl'
    print('Loading train dataset...')
    with open(train_dataset_fname, 'rb') as handle:
        train_data = pickle.load(handle)
    print('Loading val dataset...')
    with open(val_dataset_fname, 'rb') as handle:
        val_data = pickle.load(handle)
    
    print('Loading Train Dataset')
    train_dataset = CustomGNPGraspDataset(data=train_data)
    print('Loading Val Dataset')
    val_dataset = CustomGNPGraspDataset(data=val_data, context_data=train_data)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=128,
        collate_fn=custom_collate_fn,
        shuffle=True
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        collate_fn=custom_collate_fn,
        batch_size=64,
        shuffle=False
    )

    for batch in train_dataloader:
        print(f'---- {len(batch)} ----')
        for elem1 in batch:
            for elem in elem1:
                print(elem.shape)

    for batch in val_dataloader:
        print(f'---- {len(batch)} ----')
        for elem1 in batch:
            for elem in elem1:
                print(elem.shape)
