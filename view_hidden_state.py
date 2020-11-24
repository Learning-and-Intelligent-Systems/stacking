import torch
from argparse import Namespace
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from learning.train_gat import load_dataset
from learning.conv_rnn_orig import TowerConvRNNOrig

model_fname = 'results/2020-11-23_15:41:03/model1_good.pt'
test_dataset = 'random_blocks_(x1600)_2to5blocks_uniform_density_relative_pos.pkl'

# load model
model = TowerConvRNNOrig(150)
device = torch.device('cpu')
model.load_state_dict(torch.load(model_fname, map_location=device))
model.eval()

# load dataset
args = Namespace(visual=True)
test_datasets, _ = load_dataset(test_dataset, args)
dataset_2block = test_datasets[0]
dataset_3block = test_datasets[1]
dataset_4block = test_datasets[2]
dataset_5block = test_datasets[3]
dataset = dataset_5block # change which dataset to use here
dataloader = iter(DataLoader(dataset, batch_size=1, shuffle=False))
num_data_points = len(dataset)

image_dim = 150

# run through network one tower at a time
correct_preds = 0
for i in range(num_data_points):
    towers, labels, images = next(dataloader)
    # one block at a time
    h = torch.zeros(1, 1, image_dim, image_dim)
    label = labels[0].cpu().detach().numpy().squeeze()
    print('true label: ', 'stable' if label == 1. else 'unstable')
    if label == 1.:
        for block_image in torch.flip(images[0], dims=[0]):
            # view input hidden state
            fig, axes = plt.subplots(1,3)
            try:
                axes[0].imshow(h.detach().numpy().squeeze(), cmap='gray')
            except:
                axes[0].imshow(h.squeeze(), cmap='gray')
            axes[0].set_title('Input Hidden State')
            axes[0].axis('off')
            
            # view input image
            axes[1].imshow(block_image.squeeze(), cmap='gray')
            axes[1].set_title('Input Image')
            axes[1].axis('off')
            
            # run through network
            input = torch.cat([block_image.view(1,1,image_dim, image_dim), h], dim=1)
            h_small = model.encoder(input)
            
            # visualize output hidden state
            axes[2].imshow(h_small.detach().numpy().squeeze(), cmap='gray')
            axes[2].set_title('Output Hidden State')
            axes[2].axis('off')
            plt.show()
                    
            # prep next hidden state
            h = torch.zeros(1, 1, image_dim, image_dim)
            h[:,:,model.insert_h:model.insert_h+model.hidden_dim, model.insert_h:model.insert_h+model.hidden_dim] = h_small

        label = labels[0].cpu().detach().numpy().squeeze()
        pred = model.forward(images).cpu().detach().numpy().squeeze()>0.5
        if label == pred:
            correct_preds +=1
            print('correctly labeled')
print('accuracy: ', correct_preds/(i+1))
    