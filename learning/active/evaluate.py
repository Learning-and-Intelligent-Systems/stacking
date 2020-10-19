import argparse
import matplotlib.pyplot as plt
import torch

from sklearn.calibration import calibration_curve

from learning.active.utils import ExperimentLogger, get_predictions


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


def validate_confidence(logger):
    """
    Find examples where all models are confident. Check that the predictions are the correct label.
    """
    print('Checking confident examples...')

    # Get the validation dataset to test on.
    val_dataset = logger.load_dataset('val.pkl')
    ensemble = logger.get_ensemble()

    # Get predictions for each model.
    preds = get_predictions(val_dataset, ensemble)
    ys = val_dataset.ys

    # Check where all models agree.
    conf1_ix = preds.mean(dim=1) > 0.9
    conf0_ix = preds.mean(dim=1) < 0.1
    nconf1 = ys[conf1_ix].shape[0]
    nconf0 = ys[conf0_ix].shape[0]

    unconf1_ix = (preds.mean(dim=1) > 0.5) & (preds.mean(dim=1) < 0.9)
    unconf0_ix = (preds.mean(dim=1) > 0.1) & (preds.mean(dim=1) < 0.5)
    nunconf1 = ys[unconf1_ix].shape[0]
    nunconf0 = ys[unconf0_ix].shape[0]

    conf1_acc = (ys[conf1_ix] == 1).mean()
    unconf1_acc = (ys[unconf1_ix] == 1).mean()
    print('Class 1:\tConf Acc (%d): %f\tUnconf Acc (%d): %f' % (nconf1, conf1_acc, nunconf1, unconf1_acc))
    conf0_acc = (ys[conf0_ix] == 0).mean()
    unconf0_acc = (ys[unconf0_ix] == 0).mean()
    print('Class 0:\tConf Acc (%d): %f\tUnconf Acc (%d): %f' % (nconf0, conf0_acc, nunconf0, unconf0_acc))

    print(ys.shape, preds.mean(dim=1).shape)
    NBINS=10
    fraction_of_positives, mean_predicted_value = calibration_curve(ys, preds.mean(dim=1), n_bins=NBINS)
    plt.close()
    fig = plt.figure(0, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:")
    ax1.plot(mean_predicted_value, fraction_of_positives)
    ax2.hist(preds.mean(dim=1), range=(0, 1), bins=NBINS, 
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")

    plt.tight_layout()
    
    plt.savefig(logger.get_figure_path('calibration.png'))



RES = 0.005
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-path', type=str, required=True)
    args = parser.parse_args()

    logger = ExperimentLogger(args.exp_path)

    # ensemble = logger.get_ensemble()

    # all_preds = []
    # for mx, model in enumerate(ensemble):
    #     # Returns a list of the predictions for each of the dropout models.
    #     all_preds += model.plot_decision_boundary(resolution=RES, fname=logger.get_figure_path('decision_boundary_%d.png' % mx), k=1)

    # # TODO: Plot the BALD objective values for each of the contour points.
    # display_bald_objective(all_preds, resolution=RES, fname=logger.get_figure_path('bald_objective.png'))
    # display_marginal_predictions(all_preds, resolution=RES, fname=logger.get_figure_path('marginal_preds.png'))
    validate_confidence(logger)