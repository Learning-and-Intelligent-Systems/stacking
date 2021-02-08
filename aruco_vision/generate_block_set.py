import itertools
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-3.7, 3.7)
# ax.set_ylim(-10,100)
# ax.set_zlim(-10, 100)


# draw cube
def draw(dim, com, position, ax):
    signed_corners = np.array([c for c in itertools.product([-1, 1], repeat=3)])
    corners = signed_corners * dim / 2 + position
    for s, e in combinations(corners, 2):
        if ((s - e) == 0).sum() == 2:
            ax.plot3D(*zip(s, e), color="b")

    ax.scatter(*com + position, color='r')

num_blocks = 16
max_block_size = 15
com_border = 2 # how close the COM can be the edge of the block

# generate the block dimensions (in cm)
dims = np.zeros([num_blocks, 3])

for i in range(num_blocks):
    while True: 
        d = np.round((4+11*np.random.rand(3))*2)/2 
        grippable = (d < 7.5).sum() >= 2
        if grippable:
            dims[i] = d
            break 

# generate the block coms (with uniform sampling)
# coms = np.random.rand(num_blocks, 3)*(dims-com_border*2) - (dims/2) + com_border

# NOTE(izzy): blocks were uninteresting with uniform random COM. here's a different strategy:
signed_coms = np.random.randint(-1,2, size=(num_blocks, 3))
coms = signed_coms * (dims/2 - 2)

# plot the blocks
grid_size = np.ceil(np.sqrt(num_blocks))

for i, (d, c) in enumerate(zip(dims,coms)):
    print(f'Block {i}: {d}\t{c}')
    position = np.array([i%grid_size - grid_size/2, i // grid_size - grid_size/2, 0]) * max_block_size
    draw(d, c, position, ax)

plt.show()
