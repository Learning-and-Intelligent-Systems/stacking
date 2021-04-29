import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

def viz_latents(locs, scales, lim=2):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    if lim < 1:
        ax.plot([-lim, lim], [0, 0], [0, 0], c='r')
        ax.plot([0, 0], [-lim, lim], [0, 0], c='r')
        ax.plot([0, 0], [0, 0], [-lim, lim], c='r')
    ax.scatter(locs[:10, 1], 
               locs[:10, 2],
               locs[:10, 3], c=np.arange(10), cmap=cm.tab10)

    if lim > 1:
        u = np.linspace(0, 2 * np.pi, 10)
        v = np.linspace(0, np.pi, 10)
        for ix in range(10):
            x = locs[ix, 1] + scales[ix, 1] * np.outer(np.cos(u), np.sin(v))
            y = locs[ix, 2] + scales[ix, 2] * np.outer(np.sin(u), np.sin(v))
            z = locs[ix, 3] + scales[ix, 3] * np.outer(np.ones(np.size(u)), np.cos(v))

            print(type(x), x.shape)
            ax.plot_surface(x.numpy(), y.numpy(), z.numpy(), color='b', alpha=0.05)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    plt.show()


if __name__ == '__main__':
    blocks = np.load('learning/data/10_cubes_block_set2.npy')
    viz_latents(blocks[:, 0:4], blocks[:, 0:4], lim=0.035)