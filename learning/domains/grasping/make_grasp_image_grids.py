import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


MASSES = ['%.2f' % m for m in [0.1, 0.5, 1.0, 1.5, 2.0]]
FRICTIONS = ['%.2f' % f for f in [0.1, 0.5, 1.0]]


def get_image_files(args):
    fname_lookup = {}
    for m in MASSES:
        fname_lookup[m] = {}
        for f in FRICTIONS:
            fname_lookup[m][f] = []

    for f in FRICTIONS:
        for m in MASSES:
            for cx in range(0, 5):
                fname = 'images/%s_%sm_%sf_%d_%s.png' % (args.ycb_name, m, f, cx, args.viewpoint)
                fname_lookup[m][f].append(fname)

    return fname_lookup

# im1 = np.arange(100).reshape((10, 10))
# im2 = im1.T
# im3 = np.flipud(im1)
# im4 = np.fliplr(im2)


def display_images(fnames, args):
    if args.varied == 'friction':
        OUTER_ITER = MASSES
        INNER_ITER = FRICTIONS
        name = 'mass'
    else:
        OUTER_ITER = FRICTIONS
        INNER_ITER = MASSES
        name = 'friction'

    n_rows = len(INNER_ITER)
    n_cols = 5
    for out_val in OUTER_ITER:
        fig = plt.figure(figsize=(50., 50.))
        grid = ImageGrid(
            fig, 111,  # similar to subplot(111)
            nrows_ncols=(n_rows, n_cols),  # creates 2x2 grid of axes
            axes_pad=0.1,  # pad between axes in inch.
            share_all=True
        )
        print(len(grid))

        for rx, in_val in enumerate(INNER_ITER):
            
            if args.varied == 'friction':
                row_images = fnames[out_val][in_val]
            else:
                row_images = fnames[in_val][out_val]

            for cx in range(0, 5):
                image = plt.imread(row_images[cx])
                grid[rx*5+cx].imshow(image)

                grid[rx*5].set_ylabel('mass=%s' % in_val, fontsize=20)
        plt.savefig('learning/domains/grasping/images_all/%s_%s_%s.png' % (args.ycb_name, name, out_val))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--viewpoint', required=True, choices=['x', 'y', 'z'])
    parser.add_argument('--ycb-name', required=True, type=str)
    parser.add_argument('--varied', required=True, choices=['mass', 'friction'])
    args = parser.parse_args()

    images_files = get_image_files(args)
    display_images(images_files, args)


