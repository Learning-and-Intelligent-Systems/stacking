import numpy as np
import rosbag
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader


class ExtrinsicsError(nn.Module):
    def __init__(self):
        super(ExtrinsicsError, self).__init__()
        self.xyz_rpy_BC = nn.Parameter(torch.randn(6))
        self.xyz_rpy_GO = nn.Parameter(torch.randn(6))


    def euler_to_rotation_matrix(self, rpy):
        c = torch.cos
        s = torch.sin
        r, p, y = rpy
        # torch equivalent of Rotation.from_euler('xyz', theta).as_matrix()
        R = torch.Tensor([[c(p)*c(y), s(r)*s(p)*c(y) - s(y)*c(r), s(p)*c(r)*c(y) + s(r)*s(y)],
                      [s(y)*c(p), s(r)*s(p)*s(y) + c(r)*c(y), s(p)*s(y)*c(r) - s(r)*c(y)],
                      [-s(p), s(r)*c(p), c(r)*c(p)]])

        return R

    def get_pose(self, xyz_rpy):
        X = torch.eye(4)
        X[:3, :3] = self.euler_to_rotation_matrix(xyz_rpy[3:])
        X[:3, 3] = xyz_rpy[:3]
        return X
        

    def forward(self, X_BG, X_CO):
        X_GO = self.get_pose(self.xyz_rpy_GO)
        X_BC = self.get_pose(self.xyz_rpy_BC)
        return (X_BC @ X_CO - X_BG @ X_GO).pow(2).sum()

def get_data(N=100):
    from scipy.spatial.transform import Rotation
    from rotation_util import Rt_to_pose_matrix

    # synthetic data
    R_BC_ground_truth = Rotation.random().as_matrix()
    R_GO_ground_truth = Rotation.random().as_matrix()
    R_BG = Rotation.random(N).as_matrix()

    t_BC_ground_truth = np.array([1, 0, 1])
    t_GO_ground_truth = np.array([0.1, 0, 0])
    t_BG = np.random.randn(N, 3)

    X_BC_ground_truth = Rt_to_pose_matrix(R_BC_ground_truth, t_BC_ground_truth)
    X_GO_ground_truth = Rt_to_pose_matrix(R_GO_ground_truth, t_GO_ground_truth)
    X_BG = Rt_to_pose_matrix(R_BG, t_BG)
    X_CO = np.linalg.inv(X_BC_ground_truth) @ X_BG @ X_GO_ground_truth
    return X_BG, X_CO

def train(model, dataset):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    data_loader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)

    for epoch in range(5):  # loop over the dataset multiple times

        for i, (X_BG, X_CO) in enumerate(data_loader, 0):
            optimizer.zero_grad()
            loss = model(X_BG, X_CO)
            loss.backward()
            optimizer.step()

            print(f'Epoch {epoch}.{i} loss: {loss.item()}')

if __name__ == '__main__':
    X_BG, X_CO = get_data(N=1000)
    dataset = TensorDataset(torch.Tensor(X_BG), torch.Tensor(X_CO))
    model = ExtrinsicsError()
    train(model, dataset)
