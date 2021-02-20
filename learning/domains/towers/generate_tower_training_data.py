""" Massachusetts Institute of Technology

Izzy Brand, 2020
"""
from agents.teleport_agent import TeleportAgent
from block_utils import World, Environment, Object, Position, Quaternion, Pose, ZERO_POS, rotation_group, get_rotated_block, all_rotations, QUATERNIONS
from tower_planner import TowerPlanner
from pybullet_utils import transformation
import argparse
from copy import deepcopy
import numpy as np
import pickle
from random import choices as sample_with_replacement
import time
import matplotlib.pyplot as plt

def vectorize(tower):
    return [b.vectorize() for b in tower]

ROTATED_BLOCKS = {}
def sample_random_tower(blocks, num_blocks=None, ret_rotated=False, discrete=False):
    if num_blocks is None:
        num_blocks = len(blocks)

    # select blocks in this tower
    blocks = np.random.choice(blocks, num_blocks, replace=False)
    
    # pick random orientations for the blocks
    orns = sample_with_replacement(QUATERNIONS, k=num_blocks)
    #orns = [Quaternion(*orn.as_quat()) for orn in orns]
    # apply the rotations to each block
    rotated_blocks = []
    for orn, block in zip(orns, blocks):
        block.pose = Pose(ZERO_POS, orn)
        if block not in ROTATED_BLOCKS:
            ROTATED_BLOCKS[block] = {}
        if orn not in ROTATED_BLOCKS[block]:
            ROTATED_BLOCKS[block][orn] = get_rotated_block(block)
        rotated_blocks.append(ROTATED_BLOCKS[block][orn])

    # pick random positions for each block
    if discrete:
        coms = np.array([rb.com for rb in rotated_blocks])
        coms_diff = np.subtract(coms[:-1], coms[1:])
        pos_xy = np.cumsum(coms_diff[:,:2], axis=0)
        pos_xy = np.vstack([np.zeros([1,2]), pos_xy])
    else:
        # get the x and y dimensions of each block (after the rotation)
        dims_xy = np.array([rb.dimensions for rb in rotated_blocks])[:,:2]
        # figure out how far each block can be moved w/ losing contact w/ the block below
        max_displacements_xy = (dims_xy[1:] + dims_xy[:1])/2.
        # sample unscaled noise (clip bceause random normal can exceed -1, 1)
        noise_xy = np.clip(0.5*np.random.randn(num_blocks-1, 2), -0.95, 0.95)
        # and scale the noise by the max allowed displacement
        rel_xy = max_displacements_xy * noise_xy
        # place the first block at the origin
        rel_xy = np.vstack([np.zeros([1,2]), rel_xy])
        # and get the actual positions by cumulative sum of the relative positions
        pos_xy = np.cumsum(rel_xy, axis=0)

    # calculate the height of each block
    heights = np.array([rb.dimensions.z for rb in rotated_blocks])
    cumulative_heights = np.cumsum(heights)
    pos_z = heights/2
    pos_z[1:] += cumulative_heights[:-1]

    # apply the positions to each block
    pos_xyz = np.hstack([pos_xy, pos_z[:,None]])
    for pos, orn, block, rblock in zip(pos_xyz, orns, blocks, rotated_blocks):
        block.pose = Pose(Position(*pos), orn)
        block.rotation = orn
        rblock.pose = Pose(pos, (0,0,0,1))

    if ret_rotated:
        return blocks, rotated_blocks

    return blocks

def build_tower(blocks, constructable=None, stable=None, pairwise_stable=True, cog_stable=True, vis=False, max_attempts=250):
    """ Build a tower with the specified stability properties.
    :param blocks: if this is a list of blocks, use those blocks. if int, generate that many new blocks
    :param stable: Overall tower stability.
    :param pairwise_stable: The stability between two consecutive blocks in the tower.
    :param cog_stable: If the tower is stable just by looking at the CoG.
    """
   
    # init a tower planner for checking stability
    tp = TowerPlanner(stability_mode='contains')

    if isinstance(blocks, int):
        blocks = [Object.random(f'obj_{ix}') for ix in range(blocks)]

    # since the blocks are sampled with replacement from a finite set, the
    # object instances are sometimes identical. we need to deepcopy the blocks
    # one at a time to make sure that they don't share the same instance
    blocks = [deepcopy(block) for block in blocks]

    for _ in range(max_attempts):
        # generate a random tower
        tower = sample_random_tower(blocks)
        # visualize the tower if desired
        if vis: TeleportAgent.simulate_tower(tower, vis=True, T=20)
        # if the tower is stable, visualize it for debugging
        rotated_tower = [get_rotated_block(b) for b in tower]
        # save the tower if it's stable
        if not stable is None:
            if tp.tower_is_stable(rotated_tower) == stable and \
            tp.tower_is_pairwise_stable(rotated_tower) == pairwise_stable and \
            tp.tower_is_cog_stable(rotated_tower) == cog_stable: 
                return rotated_tower
        elif not constructable is None:
            if tp.tower_is_constructable(rotated_tower) == constructable and \
            tp.tower_is_pairwise_stable(rotated_tower) == pairwise_stable and \
            tp.tower_is_cog_stable(rotated_tower) == cog_stable: 
                return rotated_tower

    return None

# NOTE(caris): I think we should use the size of the dataset in the filename, not
# num_towers as sometimes 
# sum([len(dataset[num_blocks]['towers']) for num_blocks in dataset.keys()]) < num_towers
def get_filename(num_towers, use_block_set, block_set_size, suffix):
    # create a filename for the generated data based on the configuration
    block_set_string = f"{block_set_size}block_set" if use_block_set else "random_blocks"
    return f'learning/data/{block_set_string}_(x{num_towers})_{suffix}.pkl'

def generate_training_images(world):
    # NOTE (caris): these images do not capture anything about the mass of the block
    scale = .007                           # meters in a pixel
    height, width = 150, 150              # dimensions of output images
    pixel_origin = (height/2, width/2)    # pixel corresponding to world frame origin
    com_marker_width = 5                  # in pixels

    def pixel_to_world(pixel):
        x = scale*(pixel[0]-pixel_origin[0])
        y = scale*(pixel_origin[1]-pixel[1])
        return np.array([x, y])
        
        # TODD, test xy_in_obj()
    def world_to_pixel(point):
        x = pixel_origin[0] + point[0]/scale
        y = pixel_origin[1] - point[1]/scale
        return np.array([x, y])
        
    def xy_in_obj(xy, object):
        endpoints_obj = [[-object.dimensions.x/2, -object.dimensions.y/2],
                        [-object.dimensions.x/2, +object.dimensions.y/2],
                        [object.dimensions.x/2, +object.dimensions.y/2],
                        [object.dimensions.x/2, -object.dimensions.y/2]]
        endpoints_world = [transformation([epo[0], epo[1], 0.], object.pose.pos, object.pose.orn)[:2] for epo in endpoints_obj]
        line_segment_indices = [(0,1),(1,2),(2,3),(3,0)]
        inside = True
        for line_indices in line_segment_indices:
            line = [endpoints_world[index] for index in line_indices]
            line_vec = np.subtract(line[1], line[0])
            line_query = np.subtract(xy, line[0])
            if np.cross(line_vec, line_query) < 0:
                inside = inside and True
            else:
                inside = inside and False
        return inside
        
    def get_object_training_image(object):
        image = np.zeros((width, height)) #(black)
        image_xs = np.linspace(0, width, width+1).astype(np.uint32)
        image_ys = np.linspace(0, height, height+1).astype(np.uint32)
        
        # draw object (white)
        found_obj = False
        for pixel_x in image_xs:
            for pixel_y in image_ys:
                world_xy = pixel_to_world([pixel_x, pixel_y])
                if xy_in_obj(world_xy, object):
                    found_obj = True
                    image[pixel_y, pixel_x] = 1.0

        # draw COM in object (gray)
        com_world = transformation(object.com, object.pose.pos, object.pose.orn)
        com_pixel = world_to_pixel(com_world[:2])
        com_xs = np.linspace(com_pixel[0]-(com_marker_width-1)/2,
                                com_pixel[0]+(com_marker_width-1)/2,
                                com_marker_width).astype(np.uint32)
        com_ys = np.linspace(com_pixel[1]-(com_marker_width-1)/2,
                                com_pixel[1]+(com_marker_width-1)/2,
                                com_marker_width).astype(np.uint32)
        for com_x in com_xs:
            for com_y in com_ys:
                image[com_y, com_x] = 0.5

        '''
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
        '''
        return image

    images = []
    for object in world.objects:
        image = get_object_training_image(object)
        images.append(image)

    return images


def main(args, vis_tower=False):
    # This is a dictionary from stable/unstable label to what subsets of [COG_Stable, PW_Stable] to include.
    difficulty_types = {
        0: [[True, True], [True, False], [False, True], [False, False]],
        1: [[True, True], [True, False], [False, True], [False, False]]
    }

    # specify the number of towers to generate
    num_towers_per_cat = 250
    num_towers = num_towers_per_cat * 4 * 2
    # specify whether to use a finite set of blocks, or to generate new blocks
    # for each tower
    use_block_set = False
    # the number of blocks in the finite set of blocks
    block_set_size = 1000
    # generate the finite set of blocks
    if use_block_set:
        block_set = [Object.random(f'obj_{i}') for i in range(block_set_size)]

    # use_block_set = True
    # with open('learning/data/block_set_10.pkl', 'rb') as handle:
    #     block_set = pickle.load(handle)

    # create a vector of stability labels where half are unstable and half are stable
    stability_labels = np.zeros(num_towers, dtype=int)
    stability_labels[num_towers // 2:] = 1

    dataset = {}
    for num_blocks in range(2, args.max_blocks+1):
        vectorized_towers = []
        block_names = []
        images = []

        for stable in [0, 1]:
            for cog_stable, pw_stable in difficulty_types[stable]:
                # PW Stability is the same as global stability for two blocks.
                if num_blocks == 2 and pw_stable != stable:
                    continue
                elif (args.criteria == 'constructable') and pw_stable != stable:
                    continue

                count = 0
                while count < num_towers_per_cat:
                    # print the information about the tower we are about to generate
                    stability_type = "stable" if stable else "unstable"
                    stability_type += "/cog_stable" if cog_stable else "/cog_unstable"
                    stability_type += "/pw_stable" if pw_stable else "/pw_unstable"
                    print(f'{count}/{num_towers_per_cat}\t{stability_type} {num_blocks}-block tower')

                    # generate random blocks. Use the block set if specified. otherwise
                    # generate new blocks from scratch. Save the block names if using blocks
                    # from the block set
                    if use_block_set:
                        blocks = np.random.choice(block_set, num_blocks, replace=False)
                    else:
                        blocks = [Object.random(f'obj_{ix}') for ix in range(num_blocks)]

                    # generate a random tower
                    if args.criteria == 'stable':
                        tower = build_tower(blocks, 
                                            stable=stable, 
                                            pairwise_stable=pw_stable, 
                                            cog_stable=cog_stable)
                    elif args.criteria == 'constructable':
                        tower = build_tower(blocks, 
                                            constructable=stable, 
                                            pairwise_stable=pw_stable, 
                                            cog_stable=cog_stable)
                    else:
                        raise NotImplementedError()
                    
                    if tower is None:
                        continue
                    
                    # NOTE: this has to be done before the sim is run or else
                    # the images will be of the object final positions
                    if args.save_images:
                        w = World(tower)
                        training_images = generate_training_images(w)
                        images.append(training_images)

                    if vis_tower:
                        w = World(tower)
                        env = Environment([w], vis_sim=True, vis_frames=True)
                        print(stability_type)
                        input()
                        for tx in range(240):
                            env.step(vis_frames=False)
                            time.sleep(1/240.)
                        env.disconnect()
                    count += 1
                    # append the tower to the list
                    vectorized_towers.append(vectorize(tower))
                    block_names.append([b.name for b in blocks])
        if num_blocks == 2 or args.criteria == 'constructable':
            stability_labels = np.zeros(num_towers//2, dtype=int)
            stability_labels[num_towers // 4:] = 1
        else:
            stability_labels = np.zeros(num_towers, dtype=int)
            stability_labels[num_towers // 2:] = 1
        data = {
            'towers': np.array(vectorized_towers),
            'labels': stability_labels,
            'images': images
        }
        if use_block_set:
            data['block_names'] = block_names

        dataset[f'{num_blocks}block'] = data

    # save the generate data
    if args.criteria == 'constructable':
        num_towers /= 2
    filename = get_filename(num_towers, use_block_set, block_set_size, args.suffix)
    print('Saving to', filename)
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--output', type=str, required=True, help='where to save')
    parser.add_argument('--max-blocks', type=int, required=True)
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--save-images', action='store_true')
    parser.add_argument('--criteria', default='constructable', choices=['stable', 'constructible'])
    args = parser.parse_args()

    main(args, vis_tower=False)
