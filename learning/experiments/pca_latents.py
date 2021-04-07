import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

def pca_gaussians(locs, scales, return_diagonal_cov=True):
    m = locs.mean(axis=0)
    s = locs.std(axis=0)

    locs_normalized = (locs - m)/s

    cov = np.cov(locs_normalized.T)
    vals, vecs = np.linalg.eig(cov)
    a = vecs[:, np.argsort(vals)[-2:]]

    locs_proj = locs @ a
    if return_diagonal_cov: 
        scales_proj = scales @ a
    else:
        # vectorized version np.diag
        scales = np.eye(scales.shape[1])[None, ...] * scales[..., None]
        # if cov(x) = S, then cov(Ax) = A S A^T
        scales_proj = a.T @ scales @ a

    return locs_proj, scales_proj

def plot_gaussians(locs, scales):
    
    ax = plt.gca()
    for l, s in zip(locs_proj, scales_proj):
        c = np.random.rand(3)
        plt.scatter(*l, c = c)
        if s.ndim == 1:
            ax.add_patch(Ellipse(l, *s, fill=False, color=c))
        else:
            vals, vecs = np.linalg.eig(s)
            long_axis = vecs[:, np.argmax(vals)]
            angle = np.arctan2(long_axis[1], long_axis[0])
            ax.add_patch(Ellipse(l, *vals, fill=False, color=c))
    plt.title('Latent Variables without Prior.')
    plt.show()


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


    # without KL
    locs = np.array([[ 0.2029,  0.5111, -0.5805,  1.2011,  1.8219],
                     [-1.0448, -1.1561, -0.5019,  1.2886,  1.2775],
                     [-1.9459, -1.3801,  0.8102, -1.7601,  0.8340],
                     [ 1.7771,  0.0574,  1.4843, -1.4482,  0.7797],
                     [ 0.3809, -1.2340,  2.4344,  0.3318,  0.3907],
                     [ 0.6981,  0.9217, -1.2412, -0.2512, -0.5549],
                     [ 0.2590,  0.9800,  0.5843,  0.7152, -0.2053],
                     [ 1.8302, -2.4050, -0.9263,  0.2050, -2.2238],
                     [ 0.2432,  0.7225, -1.0038, -1.3842, -0.1806],
                     [ 0.1655, -2.2277,  0.1689,  2.0739,  2.3224]])
    scales = np.array([[ 0.0218,  0.0107,  0.0126, -0.0165, -0.0096],
                       [ 0.0259,  0.0090,  0.0205, -0.0238,  0.0429],
                       [ 0.0250, -0.0211,  0.0427,  0.0041, -0.0257],
                       [-0.0200,  0.0243,  0.0097,  0.0093,  0.0472],
                       [ 0.0177, -0.0027,  0.0206, -0.0166, -0.0123],
                       [ 0.0375,  0.0122, -0.0077, -0.0060,  0.0374],
                       [ 0.0363, -0.0166,  0.0400, -0.0040,  0.0046],
                       [ 0.0091,  0.0142, -0.0155,  0.0029, -0.0095],
                       [ 0.0327, -0.0089,  0.0115, -0.0027, -0.0340],
                       [-0.0067, -0.0507,  0.0305, -0.0006, -0.0017]])

    locs_proj, scales_proj = pca_gaussians(locs, scales, return_diagonal_cov=False)
    plot_gaussians(locs_proj, scales_proj)
