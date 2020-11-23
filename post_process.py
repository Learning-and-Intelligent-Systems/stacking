import torch
from torch.utils.data import DataLoader
import pickle
import matplotlib.pyplot as plt
from argparse import Namespace
import numpy as np

from pybullet_utils import transformation
from learning.cnn import TowerCNN
from learning.train_gat import load_dataset

test_dataset = 'random_blocks_(x800)_2to3blocks_uniform_density.pkl'
args = Namespace(visual=True)
test_datasets, _ = load_dataset(test_dataset, args)
batch_size = 1
dataloader = iter(DataLoader(test_datasets[0], batch_size=batch_size))

for batch_idx in range(100):
    towers, labels, images = next(dataloader)
    image = images[0][0]
    print(towers[0][0][4:7])
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
#import pdb; pdb.set_trace()

'''
test_dataset = 'random_blocks_(x2000)_2blocks_uniform_density.pkl'
model_fname = '2020-10-14_12:37:08/model.pt'

# load model
model = TowerCNN(150)
device = torch.device('cpu')
model.load_state_dict(torch.load(model_fname, map_location=device))
model.eval()

# calculate if prediction is correct for each data point
pred_correct = []
all_towers = []
args = Namespace(visual=True)
test_datasets, _ = load_dataset(test_dataset, args)
dataset = test_datasets[0]
# have to run test set through network in batches due to memory issues
num_data_points = len(dataset)
batch_size = 200
dataloader = iter(DataLoader(dataset, batch_size=batch_size))

for batch_idx in range(num_data_points // batch_size):
    towers, labels, images = next(dataloader)
    all_towers += list(towers)
    preds = model.forward(images, k=towers.shape[1]-1).squeeze()
    accuracy = (preds.cpu().detach().numpy()>0.5) == labels.cpu().detach().numpy()
    pred_correct += list(accuracy)
final_accuracy = np.mean(pred_correct)
print('Final Accuracy: ', final_accuracy)

# calc how close the top block's COM is to the edge of the bottom block
min_dists = []
coms_world = [transformation(tower[1][1:4], tower[1][7:10], tower[1][10:14]) for tower in all_towers]
for com, bot_block in zip(coms_world, [tower[0] for tower in all_towers]):
    endpoints_obj = [[-bot_block[4]/2, -bot_block[5]/2],
                    [-bot_block[4]/2, bot_block[5]/2],
                    [bot_block[4]/2, bot_block[5]/2],
                    [bot_block[4]/2, -bot_block[5]/2]]
    endpoints_world = [transformation([epo[0], epo[1], 0.], bot_block[7:10], bot_block[10:14])[:2] for epo in endpoints_obj]
    line_segment_indices = [(0,1),(1,2),(2,3),(3,0)]
    distances = []
    for line_indices in line_segment_indices:
        line = [endpoints_world[index] for index in line_indices]
        line_length = np.linalg.norm(line[1]-line[0])
        distance = abs((line[1][1]-line[0][1])*com[0] - (line[1][0]-line[0][0])*com[1] + line[1][0]*line[0][1] - line[1][1]*line[0][0])/line_length
        distances.append(distance)
    min_dists.append(min(distances))

'''
'''
# plot block params (width, length, height, mass, stability label, versus accuracy
top_masses = [tower[1][0] for tower in all_towers]
top_dims_x = [tower[1][4] for tower in all_towers]
top_dims_y = [tower[1][5] for tower in all_towers]
top_dims_z = [tower[1][6] for tower in all_towers]

fig, ax = plt.subplots()
ax.plot(top_masses, pred_correct, '.')
ax.set_title('Mass')

fig, ax = plt.subplots()
ax.plot(top_dims_x, pred_correct, '.')
ax.set_title('X Dim')

fig, ax = plt.subplots()
ax.plot(top_dims_y, pred_correct, '.')
ax.set_title('Y Dim')

fig, ax = plt.subplots()
ax.plot(top_dims_z, pred_correct, '.')
ax.set_title('Z Dim')

fig, ax = plt.subplots()
ax.plot(min_dists, pred_correct, '.')
ax.set_title('COM Dist to Edge')
'''
'''
fig, ax = plt.subplots()
ax.hist(min_dists, bins=50)
ax.set_title('Histogram of Distances to Edge')

fig, ax = plt.subplots()
failed_min_dists = [min_dist for (min_dist, pred) in zip(min_dists, pred_correct) if not pred]
ax.hist(failed_min_dists)
ax.set_title('Failed Predictions Distances to Edge')

plt.show()
'''    