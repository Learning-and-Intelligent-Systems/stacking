import argparse
import pickle
import torch

from torch.nn import functional as F
from torch.utils.data import DataLoader

from learning.models.grasp_np.dataset import CustomGNPGraspDataset, custom_collate_fn
from learning.models.grasp_np.grasp_neural_process import CustomGraspNeuralProcess

# TODO: Save best models.

def get_accuracy(y_probs, target_ys):
    assert(y_probs.shape == target_ys.shape)
    acc =  ((y_probs > 0.5) == target_ys).float().mean()
    return acc


def get_loss(y_probs, target_ys, q_z, alpha=1):
    bce_loss = F.binary_cross_entropy(y_probs.squeeze(), target_ys.squeeze(), reduction='sum')

    p_z = torch.distributions.normal.Normal(torch.zeros_like(q_z.loc), torch.ones_like(q_z.scale))
    kld_loss = torch.distributions.kl_divergence(q_z, p_z).sum()
    kld_loss = 0
    # weight = (1 + alpha)
    return bce_loss + kld_loss, bce_loss, kld_loss

def train(train_dataloader, train_dataloader_val, val_dataloader_val, model, n_epochs=10):
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_loss = 10000
    alpha = 0.
    for ep in range(n_epochs):
        print(f'----- Epoch {ep} -----')
        alpha *= 0.75
        epoch_loss, train_probs, train_targets  = 0, [], []
        model.train()
        for bx, (context_data, target_data) in enumerate(train_dataloader):
            c_grasp_geoms, c_midpoints, c_labels = context_data
            t_grasp_geoms, t_midpoints, t_labels = target_data
            if torch.cuda.is_available():
                c_grasp_geoms, c_midpoints, c_labels = c_grasp_geoms.cuda(), c_midpoints.cuda(), c_labels.cuda()
                t_grasp_geoms, t_midpoints, t_labels = t_grasp_geoms.cuda(), t_midpoints.cuda(), t_labels.cuda()

            optimizer.zero_grad()

            y_probs, q_z = model.forward((c_grasp_geoms, c_midpoints, c_labels), (t_grasp_geoms, t_midpoints))
            n_predictions = target_ys.shape[1]
            y_probs = y_probs.squeeze()[:, :n_predictions]
            
            loss, bce_loss, kld_loss = get_loss(y_probs, target_ys, q_z, alpha=alpha)
            #if bx % 200 == 0:
            #    print(q_z.loc[0:5,...], q_z.scale[0:5,...])
            #    print(f'Loss: {loss.item()}\tBCE: {bce_loss}\tKLD: {kld_loss}')

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            train_probs.append(y_probs.flatten())
            train_targets.append(target_ys.flatten())

            
        epoch_loss /= len(train_dataloader.dataset)
        train_acc = get_accuracy(torch.cat(train_probs).flatten(), torch.cat(train_targets).flatten())

        # print(f'Train Loss: {epoch_loss}\tTrain Acc: {train_acc}')

        model.eval()
        val_loss, val_probs, val_targets = 0, [], []
        with torch.no_grad():
            for bx, (context_data, target_data) in enumerate(train_dataloader_val):
                c_grasp_geoms, c_midpoints, c_labels = context_data
                t_grasp_geoms, t_midpoints, t_labels = target_data
                if torch.cuda.is_available():
                    c_grasp_geoms, c_midpoints, c_labels = c_grasp_geoms.cuda(), c_midpoints.cuda(), c_labels.cuda()
                    t_grasp_geoms, t_midpoints, t_labels = t_grasp_geoms.cuda(), t_midpoints.cuda(), t_labels.cuda()
                y_probs, q_z = model.forward((c_grasp_geoms, c_midpoints, c_labels), (t_grasp_geoms, t_midpoints))
                n_predictions = target_ys.shape[1]
                y_probs = y_probs.squeeze()[:, :n_predictions]
                
                if bx % 200 == 0:
                    print(q_z.loc[0:5,...], q_z.scale[0:5,...])
                val_loss += get_loss(y_probs, target_ys, q_z)[0].item()

                val_probs.append(y_probs.flatten())
                val_targets.append(target_ys.flatten())
            
            val_loss /= len(train_dataloader_val.dataset)
            val_acc = get_accuracy(torch.cat(val_probs), torch.cat(val_targets))
            print(f'Train Loss: {val_loss}\tTrain Acc: {val_acc}')

        val_loss, val_probs, val_targets = 0, [], []
        with torch.no_grad():
            for bx, (contexts, target_xs, target_ys) in enumerate(val_dataloader_val):
                c_grasp_geoms, c_midpoints, c_labels = context_data
                t_grasp_geoms, t_midpoints, t_labels = target_data
                if torch.cuda.is_available():
                    c_grasp_geoms, c_midpoints, c_labels = c_grasp_geoms.cuda(), c_midpoints.cuda(), c_labels.cuda()
                    t_grasp_geoms, t_midpoints, t_labels = t_grasp_geoms.cuda(), t_midpoints.cuda(), t_labels.cuda()
                y_probs, q_z = model.forward((c_grasp_geoms, c_midpoints, c_labels), (t_grasp_geoms, t_midpoints))
                n_predictions = target_ys.shape[1]
                y_probs = y_probs.squeeze()[:, :n_predictions]
                
                val_loss += get_loss(y_probs, target_ys, q_z)[0].item()

                val_probs.append(y_probs.flatten())
                val_targets.append(target_ys.flatten())
            
            val_loss /= len(val_dataloader_val.dataset)
            val_acc = get_accuracy(torch.cat(val_probs), torch.cat(val_targets))
            print(f'Val Loss: {val_loss}\tVal Acc: {val_acc}')

def print_dataset_stats(dataset, name):
    print(f'----- {name} Dataset Statistics -----')
    print(f'N: {len(dataset)}')
    print(f'Context Shape: {dataset.contexts[0].shape}')
    print(f'Target xs Shape: {dataset.target_xs[0].shape}')
    print(f'Target ys Shape: {dataset.target_xs[0].shape}')

if __name__ == '__main__':

    train_dataset_fname = 'learning/data/grasping/train-sn100-test-sn10-robust-large-gnp/grasps/training_phase/train_grasps.pkl'
    val_dataset_fname = 'learning/data/grasping/train-sn100-test-sn10-robust-large-gnp/grasps/training_phase/val_grasps.pkl'
    print('Loading train dataset...')
    with open(train_dataset_fname, 'rb') as handle:
        train_data_large = pickle.load(handle)
    print('Loading val dataset...')
    with open(val_dataset_fname, 'rb') as handle:
        val_data_large = pickle.load(handle)
    
    train_dataset_fname = 'learning/data/grasping/train-sn100-test-sn10-robust-gnp/grasps/training_phase/train_grasps.pkl'
    val_dataset_fname = 'learning/data/grasping/train-sn100-test-sn10-robust-gnp/grasps/training_phase/val_grasps.pkl'
    print('Loading train dataset...')
    with open(train_dataset_fname, 'rb') as handle:
        train_data_small = pickle.load(handle)
    print('Loading val dataset...')
    with open(val_dataset_fname, 'rb') as handle:
        val_data_small = pickle.load(handle)

    train_dataset = CustomGNPGraspDataset(data=train_data_large)
    train_dataset_val = CustomGNPGraspDataset(data=val_data_large, context_data=train_data_large)
    val_dataset_val = CustomGNPGraspDataset(data=val_data_small, context_data=train_data_small)


    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=16,
        collate_fn=custom_collate_fn,
        shuffle=True
    )
    train_dataloader_val = DataLoader(
        dataset=train_dataset_val,
        batch_size=16,
        collate_fn=custom_collate_fn,
        shuffle=False
    )
    val_dataloader_val = DataLoader(
        dataset=val_dataset_val,
        collate_fn=custom_collate_fn,
        batch_size=16,
        shuffle=False
    )

    model = CustomGraspNeuralProcess(d_latents=5)

    train(train_dataloader=train_dataloader,
        train_dataloader_val=train_dataloader_val,
        val_dataloader_val=val_dataloader_val,
        model=model,
        n_epochs=1000
    )


