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
        return (X_BC @ X_CO - X_BG @ X_GO).pow(2).mean()


def train(model, dataset):
    best_loss = None
    best_pose = None
    for _ in range(100):
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
        data_loader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)
        model.reset()

        loss = 0
        for epoch in range(5):  # loop over the dataset multiple times

            for i, (X_BG, X_CO) in enumerate(data_loader, 0):
                optimizer.zero_grad()
                loss = model(X_BG, X_CO)
                loss.backward()
                optimizer.step()

        print(f'Final Loss: {loss.item()}')
        if best_loss is None:
            best_loss = loss
        elif loss < best_loss:
            best_loss = loss
            best_poses = (model.xyz_rpy_BC.detach(), model.xyz_rpy_GO.detach())
            # print(f'Epoch {epoch}.{i} loss: {loss.item()}')

    print(best_poses)

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
    X_BG, X_CO = load_rosbag_data('aruco_vision/cal.bag')
    print(X_BG.shape)
    dataset = TensorDataset(torch.Tensor(X_BG), torch.Tensor(X_CO))
    model = ExtrinsicsError()
    train(model, dataset)
