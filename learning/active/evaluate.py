import argparse
import matplotlib.pyplot as plt
import torch

from learning.active.utils import ExperimentLogger


def display_marginal_predictions(all_preds, resolution, fname):
    eps = 1e-5
    all_preds = torch.stack(all_preds)

    p = torch.mean(all_preds, dim=0)

    x1 = torch.arange(-1, 1, resolution)
    x2 = torch.arange(-1, 1, resolution)
    x1s, x2s = torch.meshgrid(x1, x2)
    K = x1s.shape[0]
    p = p.view(K, K)
    plt.close()
    plt.pcolormesh(x1s.numpy(), x2s.numpy(), p.numpy())
    plt.savefig(fname)


def display_bald_objective(all_preds, resolution, fname):
    """
    :param all_preds: A list of predictions for each model.
    :return: A tensor of the BALD value for each predicted point.
    """
    print(all_preds[0].shape)
    eps = 1e-5
    all_preds = torch.stack(all_preds)

    mp_c1 = torch.mean(all_preds, dim=0)
    mp_c0 = torch.mean(1 - all_preds, dim=0)

    m_ent = -(mp_c1 * torch.log(mp_c1+eps) + mp_c0 * torch.log(mp_c0+eps))

    p_c1 = all_preds
    p_c0 = 1 - all_preds
    ent_per_model = p_c1 * torch.log(p_c1+eps) + p_c0 * torch.log(p_c0+eps)
    ent = torch.mean(ent_per_model, dim=0)

    bald = m_ent + ent

    x1 = torch.arange(-1, 1, resolution)
    x2 = torch.arange(-1, 1, resolution)
    x1s, x2s = torch.meshgrid(x1, x2)
    K = x1s.shape[0]
    bald = bald.view(K, K)
    plt.close()
    plt.pcolormesh(x1s.numpy(), x2s.numpy(), bald.numpy())
    plt.savefig(fname)


RES = 0.005
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-path', type=str, required=True)
    args = parser.parse_args()

    logger = ExperimentLogger(args.exp_path)

    ensemble = logger.get_ensemble()

    all_preds = []
    for mx, model in enumerate(ensemble):
        # Returns a list of the predictions for each of the dropout models.
        all_preds += model.plot_decision_boundary(resolution=RES, fname=logger.get_figure_path('decision_boundary_%d.png' % mx), k=1)

    # TODO: Plot the BALD objective values for each of the contour points.
    display_bald_objective(all_preds, resolution=RES, fname=logger.get_figure_path('bald_objective.png'))
    display_marginal_predictions(all_preds, resolution=RES, fname=logger.get_figure_path('marginal_preds.png'))