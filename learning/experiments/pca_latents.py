import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse


def pca(X, d=2):
    m = X.mean(axis=0)
    s = X.std(axis=0)
    X = (X - m)/s
    cov = np.cov(X.T)
    vals, vecs = np.linalg.eig(cov)
    return vecs[:, np.argsort(vals)[-d:]]

def project_gaussians(A, locs, scales, return_diagonal_cov=True):
    locs_proj = locs @ A
    if return_diagonal_cov: 
        scales_proj = scales @ A
    else:
        # vectorized version np.diag
        scales = np.eye(scales.shape[1])[None, ...] * scales[..., None]
        # if cov(x) = S, then cov(Ax) = A S A^T
        scales_proj = A.T @ scales @ A

    return locs_proj, scales_proj

def pca_gaussians(locs, scales, return_diagonal_cov=True):
    A = pca(locs)
    return project_gaussians(A,locs, scales,
        return_diagonal_cov=return_diagonal_cov)

def plot_gaussians(locs, scales, ax=None, colors=None, alpha=1, show=True):
    if ax is None: ax = plt.gca()
    for i, (l, s) in enumerate(zip(locs, scales)):
        c = np.random.rand(3) if colors is None else colors[i]
        ax.scatter(*l, color=c, alpha=alpha)
        if s.ndim == 1:
            ax.add_patch(Ellipse(l, *s, fill=False, color=c, alpha=alpha))
        else:
            vals, vecs = np.linalg.eig(s)
            long_axis = vecs[:, np.argmax(vals)]
            angle = np.arctan2(long_axis[1], long_axis[0])
            ax.add_patch(Ellipse(l, *vals, fill=False, color=c, alpha=alpha))
    if show: plt.show()


if __name__ == '__main__':
    # with KL
    # locs = np.array([[-0.9791, -0.4076,  0.5230,  1.1171,  1.1084],
    #                  [ 1.7399, -0.7979,  0.8917,  1.4884,  0.6852],
    #                  [ 0.5745,  2.3895,  0.2699,  0.5310, -0.1292],
    #                  [-0.1726,  0.8365, -2.3493, -0.5674,  1.0960],
    #                  [ 1.5964,  0.6357,  0.1291, -0.8031,  1.2131],
    #                  [-0.9302, -0.5315, -0.2928, -0.4217, -1.0993],
    #                  [-0.2375, -0.0179, -0.3481, -0.6189,  0.5131],
    #                  [ 0.8826, -2.0611, -0.0304, -2.6109, -0.7533],
    #                  [-0.7666,  0.7222, -0.7291, -0.7975, -0.5875],
    #                  [ 1.5137, -0.4237,  2.0679,  0.1678,  3.1415]])
    # scales = np.array([[0.3319, 0.1364, 0.2730, 0.3113, 0.2754],
    #                    [0.3087, 0.2020, 0.3031, 0.2504, 0.2308],
    #                    [0.3005, 0.2060, 0.3096, 0.2497, 0.2776],
    #                    [0.3027, 0.1584, 0.2608, 0.2558, 0.2175],
    #                    [0.3190, 0.1598, 0.3076, 0.2254, 0.1995],
    #                    [0.3783, 0.2047, 0.2962, 0.2558, 0.2980],
    #                    [0.3155, 0.1987, 0.3093, 0.2917, 0.2500],
    #                    [0.3249, 0.1966, 0.3439, 0.2681, 0.2980],
    #                    [0.3235, 0.1955, 0.3585, 0.2812, 0.2036],
    #                    [0.3167, 0.1881, 0.2241, 0.2705, 0.2029]])

    # locs_proj, scales_proj = pca_gaussians(locs, scales, return_diagonal_cov=False)
    # plot_gaussians(locs_proj, scales_proj)

    # plot the latents over time
    data = np.load('learning/experiments/logs/latents/fit_during_test.npy')[:50]
    locs, scales = np.split(data, 2, axis=2)
    A = pca(locs[-1])
    ax = plt.gca()
    colors = np.random.rand(10,3)
    alphas = np.linspace(0.2, 0.2, locs.shape[0])
    for i, (l, s) in enumerate(zip(locs, scales)):
        l, s = project_gaussians(A, l, s)
        plot_gaussians(l, s, colors=colors, alpha=alphas[i], show=False)
    plt.show()
