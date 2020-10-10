import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import Dataset, DataLoader

class ToyDataGenerator:
    """
    Class to generate a low dimensional dataset with a complicated
    decision boundary.
    """
    def __init__(self):
        self.xrange = [-1., 1.]
        self.yrange = [-1., 1.]
        self.alpha = 8
        self.amp = 0.25

    def get_labels(self, data):
        """
        :param data: N x 2 numpy array of x, y coordinates.
        """
        labels = np.zeros((data.shape[0],), dtype='float32')

        labels[(data[:, 1] > 0.75) & (data[:, 0] >= 0)] = 1
        labels[((data[:, 1] > -0.25) & (data[:, 1] < 0.25)) & (data[:, 0] >= 0)] = 1
        labels[(data[:, 1] < -0.75) & (data[:, 0] >= 0)] = 1

        sin_vals = self.amp*np.sin(self.alpha*np.pi*data[:, 1])
        labels[(data[:, 1] >= 0.25) & (data[:, 1] <= 0.75) & (data[:, 0] > sin_vals)] = 1
        labels[(data[:, 1] >= -0.75) & (data[:, 1] <= -0.25) & (data[:, 0] > sin_vals)] = 1
        
        return labels


    def decision_boundary(self, resolution):
        xs, ys = np.meshgrid(np.arange(self.xrange[0], self.xrange[1], resolution),
                             np.arange(self.yrange[0], self.yrange[1], resolution))
        zs = np.zeros(xs.shape)

        zs[(ys > 0.75) & (xs >= 0)] = 1
        zs[((ys > -0.25) & (ys < 0.25)) & (xs >= 0)] = 1
        zs[(ys < -0.75) & (xs >= 0)] = 1

        sin_vals = self.amp*np.sin(self.alpha*np.pi*ys)
        zs[(ys >= 0.25) & (ys <= 0.75) & (xs > sin_vals)] = 1
        zs[(ys >= -0.75) & (ys <= -0.25) & (xs > sin_vals)] = 1

        return xs, ys, zs

    def generate_uniform_dataset(self, N):
        dataset = np.random.uniform(low=-1., high=1., size=(N, 2))
        dataset = np.float32(dataset)
        labels = self.get_labels(dataset)
        return dataset, labels

    def plot_decision_boundary(self):
        resolution = 0.005
        colors = {0: 'r', 1: 'g'}

        xs, ys, labels = self.decision_boundary(resolution)
        plt.pcolormesh(xs, ys, labels)

        plt.xlim(*self.xrange)
        plt.ylim(*self.yrange)
        plt.show()

    def plot_dataset(self, dataset, labels, fname=''):
        plt.close()
        plt.scatter(dataset[:, 0], dataset[:, 1], s=2, c=labels)
        if len(fname) > 0:
            plt.savefig(fname)
        else:
            plt.show()


class ToyDataset(Dataset):
    """
    PyTorch dataset interface for training a NN.
    """
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys


    def __len__(self):
        return self.xs.shape[0]

    def __getitem__(self, ix):
        return self.xs[ix, :], self.ys[ix]



if __name__ == '__main__':
    # TODO: Generate the dataset.
    generator = ToyDataGenerator()
    generator.plot_decision_boundary()

    dataset, labels = generator.generate_uniform_dataset(1000)
    generator.plot_dataset(dataset, labels)
    
    torch_dataset = ToyDataset(dataset, labels)
    dataloader = DataLoader(torch_dataset,
                            batch_size=32,
                            shuffle=True)

    