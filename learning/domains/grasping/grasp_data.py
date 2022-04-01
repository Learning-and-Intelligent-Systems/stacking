from torch.utils.data import Dataset, DataLoader, Sampler

class GraspDataset(Dataset):
    def __init__(self, data):
        self.grasp_vectors, self.grasp_labels = data

    def __getitem__(self, ix):
        """
        Return a DxN tensor.
        """
        return (self.grasp_vectors[ix].T, self.grasp_labels[ix])

    def __len__(self):
        return len(self.grasp_vectors)

    def add_to_dataset(self):
        pass

        

