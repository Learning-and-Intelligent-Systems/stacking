import numpy as np
import rosbag
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

import pb_robot

class ExtrinsicsError(nn.Module):
    def __init__(self):
        super(ExtrinsicsError, self).__init__()
        self.reset()

    def reset(self):
        self.xyz_rpy_BC = nn.Parameter(torch.randn(6))
        self.xyz_rpy_GO = nn.Parameter(torch.randn(6)/100)

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
        X_BCO = torch.einsum('ij,njk->nik', X_BC, X_CO)
        X_BGO = torch.einsum('nij,jk->nik', X_BG, X_GO)
        return (X_BCO - X_BGO).pow(2).mean()


def train(model, dataset):
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    data_loader = DataLoader(dataset=dataset, batch_size=128, shuffle=True)
    loss = 0
    for epoch in range(10):  # loop over the dataset multiple times

        for i, (X_BG, X_CO) in enumerate(data_loader, 0):
            optimizer.zero_grad()
            loss = model(X_BG, X_CO)
            loss.backward()
            optimizer.step()

    return loss.item()




def get_synthetic_data(N=100):
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

def object_pose_message_to_X_CO(message):
    raw_text = message.message.data
    matrix_text = raw_text.replace('[', '').replace(']', '').split(' at ')[1]
    return np.array([[float(e) for e in row.split()] for row in matrix_text.split('\n')])

def load_rosbag_data(filename):
    # get a panda for forward kinematics
    pb_robot.utils.connect(use_gui=False)
    panda = pb_robot.panda.Panda()

    X_COs = []
    X_BGs = []
    prev_joint_states_message = None

    bag = rosbag.Bag(filename, 'r')
    for message in bag.read_messages():
        if message.topic == '/object_pose':
            if prev_joint_states_message is None: continue
            X_CO = object_pose_message_to_X_CO(message)
            X_BG = panda.arm.ComputeFK(prev_joint_states_message.message.position[:7])
            time_difference = np.abs(message.timestamp.to_sec() \
                - prev_joint_states_message.timestamp.to_sec())

            if time_difference < 0.01:
                X_COs.append(X_CO)
                X_BGs.append(X_BG)

        elif message.topic == '/joint_states':
            prev_joint_states_message = message

    bag.close()
    pb_robot.utils.disconnect()
    return np.array(X_BGs), np.array(X_COs)



if __name__ == '__main__':
    # X_BG, X_CO = get_synthetic_data(N=1000)
    print('Loading rosbag')
    X_BG, X_CO = load_rosbag_data('aruco_vision/cal.bag')
    print(f'Got {X_BG.shape[0]} pairs.')
    dataset = TensorDataset(torch.Tensor(X_BG), torch.Tensor(X_CO))
    model = ExtrinsicsError()


    best_loss = None
    best_poses = None
    num_tries = 5000
    for i in range(num_tries):
        model.reset()
        loss = train(model, dataset)

        print(f'Loss {i+1}/{num_tries}: {loss}')
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_poses = (model.xyz_rpy_BC.detach(), model.xyz_rpy_GO.detach())
            print(f'New best! {best_loss}')

    print(best_loss, best_poses)