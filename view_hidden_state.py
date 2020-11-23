import torch
from argparse import Namespace
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from learning.train_gat import load_dataset
from learning.conv_rnn import TowerConvRNN

model_fname = 'results/2020-10-20_01:38:17/model.pt'
test_dataset = 'random_blocks_(x800)_2blocks_uniform_density.pkl'

# load model
model = TowerConvRNN(150)
device = torch.device('cpu')
model.load_state_dict(torch.load(model_fname, map_location=device))
model.eval()

# load dataset


# calculate if prediction is correct for each data point
args = Namespace(visual=True)
test_datasets, _ = load_dataset(test_dataset, args)
dataset = test_datasets[0]
dataloader = iter(DataLoader(dataset, batch_size=1, shuffle=True))
num_data_points = len(dataset)

image_dim = 150

# run through network one tower at a time
for i in range(num_data_points):
    towers, labels, images = next(dataloader)
    # one block at a time
    h = torch.zeros(1, 1, image_dim, image_dim)
    if labels[0].squeeze() == 0:
        pass
    else:
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
            axes[2].set_title('Input Hidden State')
            axes[2].axis('off')
            plt.show()
            
            print('true: ', labels[0])
            print('red: ', model.forward(images))
                    
            # prep next hidden state
            h = torch.zeros(1, 1, image_dim, image_dim)
            h[:,:,model.insert_h:model.insert_h+model.hidden_dim, model.insert_h:model.insert_h+model.hidden_dim] = h_small

        