import argparse
import pickle
import torch

from torch.nn import functional as F
from torch.utils.data import DataLoader

from learning.models.grasp_np.dataset import GNPGraspDataset
from learning.models.grasp_np.grasp_neural_process import GraspNeuralProcess

# TODO: Get accuracies.
# TODO: Save best models.

def get_accuracy(y_probs, target_ys):
    assert(y_probs.shape == target_ys.shape)
    acc =  ((y_probs > 0.5) == target_ys).float().mean()
    return acc


def get_loss(y_probs, target_ys, q_z):
    bce_loss = F.binary_cross_entropy(y_probs.squeeze(), target_ys, reduction='sum')

    p_z = torch.distributions.normal.Normal(torch.zeros_like(q_z.loc), torch.ones_like(q_z.scale))
    kld_loss = torch.distributions.kl_divergence(q_z, p_z).sum()
    
    return bce_loss + kld_loss

def train(train_dataloader, val_dataloader, model, n_epochs=10):

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_loss = 10000
    for ep in range(n_epochs):
        print(f'----- Epoch {ep} -----')
        epoch_loss, train_probs, train_targets  = 0, [], []
        model.train()
        for bx, (contexts, target_xs, target_ys) in enumerate(train_dataloader):
            optimizer.zero_grad()

            y_probs, q_z = model.forward(contexts, target_xs)
            loss = get_loss(y_probs, target_ys, q_z)
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            train_probs.append(y_probs.squeeze())
            train_targets.append(target_ys)
            

            if bx > 10: break
        epoch_loss /= len(train_dataloader.dataset)
        train_acc = get_accuracy(torch.cat(train_probs), torch.cat(train_targets))

        print(f'Train Loss: {epoch_loss}\tTrain Acc: {train_acc}')

        model.eval()
        val_loss, val_probs, val_targets = 0, [], []
        with torch.no_grad():
            for bx, (contexts, target_xs, target_ys) in enumerate(val_dataloader):
                y_probs, q_z = model.forward(contexts, target_xs, n_context=50)
                val_loss += get_loss(y_probs, target_ys, q_z).item()

                val_probs.append(y_probs.squeeze())
                val_targets.append(target_ys)
                if bx > 10: break
            val_loss /= len(val_dataloader.dataset)
            val_acc = get_accuracy(torch.cat(val_probs), torch.cat(val_targets))
            print(f'Val Loss: {val_loss}\tVal Acc: {val_acc}')


def print_dataset_stats(dataset, name):
    print(f'----- {name} Dataset Statistics -----')
    print(f'N: {len(dataset)}')
    print(f'Context Shape: {dataset.contexts[0].shape}')
    print(f'Target xs Shape: {dataset.target_xs[0].shape}')
    print(f'Target ys Shape: {dataset.target_xs[0].shape}')

if __name__ == '__main__':

    train_dataset_fname = 'learning/data/grasping/train-sn100-test-sn10-robust/grasps/training_phase/train_grasps.pkl'
    val_dataset_fname = 'learning/data/grasping/train-sn100-test-sn10-robust/grasps/training_phase/val_grasps.pkl'
    print('Loading train dataset...')
    with open(train_dataset_fname, 'rb') as handle:
        train_data = pickle.load(handle)
    print('Loading val dataset...')
    with open(val_dataset_fname, 'rb') as handle:
        val_data = pickle.load(handle)
    
    train_dataset = GNPGraspDataset(data=train_data)
    val_dataset = GNPGraspDataset(data=val_data, context_data=train_data)
    print_dataset_stats(train_dataset, 'Train')
    print_dataset_stats(val_dataset, 'Val')

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=16,
        shuffle=True
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=16,
        shuffle=False
    )

    model = GraspNeuralProcess(d_latents=5)

    train(train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model=model
    )


