import argparse
import json
import pickle
import torch

from sklearn.metrics import recall_score, precision_score, accuracy_score
from torch.utils.data import DataLoader

from learning.active.utils import ActiveExperimentLogger
from learning.domains.grasping.grasp_data import GraspDataset


def get_validation_metrics(logger, val_dataset_fname):
    # TODO: Load dataset
    with open(val_dataset_fname, 'rb') as handle:
        val_data = pickle.load(handle)
    val_dataset = GraspDataset(val_data)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    model = logger.get_ensemble(0)
    if torch.cuda.is_available:
        model = model.cuda()
    model.eval()

    # TODO: Get predictions
    predictions, labels = [], []
    for x, y in val_dataloader:
        if torch.cuda.is_available():
            x = x.float().cuda()
        with torch.no_grad():
            probs = model.forward(x).mean(dim=1).cpu()

        preds = (probs > 0.5).float()
        
        predictions.append(preds)
        labels.append(y)

    predictions = torch.cat(predictions).numpy()
    labels = torch.cat(labels).numpy()

    # TODO: Calculate metrics
    metrics_fn = {
        'Accuracy': accuracy_score,
        'Precision': precision_score,
        'Recall': recall_score
    } 

    metrics_val = {}
    for name, fn in metrics_fn.items():
        metrics_val[name] = fn(labels, predictions)
    
    with open(logger.get_figure_path('metrics.json'), 'w') as handle:
        json.dump(metrics_val, handle)
    print(metrics_val)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-path', type=str, required=True)
    parser.add_argument('--val-dataset-fname', type=str, required=True)
    args = parser.parse_args()

    logger = ActiveExperimentLogger(args.exp_path, use_latents=False)

    get_validation_metrics(logger, args.val_dataset_fname)