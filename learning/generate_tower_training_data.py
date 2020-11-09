""" Massachusetts Institute of Technology

Izzy Brand, 2020
"""
from agents.teleport_agent import TeleportAgent
from block_utils import World, Environment, Object, Quaternion, Pose, ZERO_POS, \
    rotation_group, get_rotated_block, Position, Quaternion, Dimensions
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

def sample_random_tower(blocks):
    num_blocks = len(blocks)
    # pick random orientations for the blocks
    orns = sample_with_replacement(list(rotation_group()), k=num_blocks)
    orns = [Quaternion(*orn.as_quat()) for orn in orns]
    # apply the rotations to each block
    rotated_blocks = []
    for orn, block in zip(orns, blocks):
        block.pose = Pose(ZERO_POS, orn)
        rotated_blocks.append(get_rotated_block(block))

    # pick random positions for each block
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
    for pos, orn, block in zip(pos_xyz, orns, blocks):
        block.pose = Pose(pos, orn)

    return blocks

def build_tower(blocks, stable=True, pairwise_stable=True, cog_stable=True, vis=False, max_attempts=250):
    """ Build a tower with the specified stability properties.
    :param stable: Overall tower stability.
    :param pairwise_stable: The stability between two consecutive blocks in the tower.
    :param cog_stable: If the tower is stable just by looking at the CoG.
    """
   
    # init a tower planner for checking stability
    tp = TowerPlanner(stability_mode='contains')

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
        if tp.tower_is_stable(rotated_tower) == stable and \
           tp.tower_is_constructible(rotated_tower) == pairwise_stable: #and \
           #tp.tower_is_cog_stable(rotated_tower) == cog_stable: 
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
    scale = .0035                           # meters in a pixel
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
        
    def add_object(image):
        endpoints_obj = [[-object.dimensions.x/2, -object.dimensions.y/2],
                        [-object.dimensions.x/2, +object.dimensions.y/2],
                        [object.dimensions.x/2, +object.dimensions.y/2],
                        [object.dimensions.x/2, -object.dimensions.y/2]]
        endpoints_world = [transformation([epo[0], epo[1], 0.], object.pose.pos, object.pose.orn)[:2] for epo in endpoints_obj]
        pixel_endpoints = [world_to_pixel(endpoint) for endpoint in endpoints_world]
        min_x_pixel = int(min([x for (x,y) in pixel_endpoints]))
        max_x_pixel = int(max([x for (x,y) in pixel_endpoints]))
        min_y_pixel = int(min([y for (x,y) in pixel_endpoints]))
        max_y_pixel = int(max([y for (x,y) in pixel_endpoints]))
        if min_x_pixel < 0 or max_x_pixel > width or min_y_pixel < 0 or max_y_pixel > height:
            return None, False
            #raise Exception('Object is at the edge of the image! Increase scale and try again.')
        else:
            image[min_y_pixel:max_y_pixel, min_x_pixel:max_x_pixel] = 1.0
            return image, True

    def add_object_rot(image):
        image_xs = np.linspace(0, width-1, width).astype(np.uint32)
        image_ys = np.linspace(0, height-1, height).astype(np.uint32)
        found_obj = False
        for pixel_x in image_xs:
            for pixel_y in image_ys:
                world_xy = pixel_to_world([pixel_x, pixel_y])
                if xy_in_obj(world_xy, object):
                    found_obj = True
                    image[pixel_y, pixel_x] = 1.0
    
    def get_object_training_image(object):
        image = np.zeros((width, height)) #(black)
        
        # draw object (white)
        all_zero_rot = True # all block orn = (0,0,0,1)
        if all_zero_rot:
            image, success = add_object(image)
        else: # TODO: need way to detect if it's at the edge
            add_object_rot(image, width, height)
            success = True
        
        # draw COM in object (gray)
        com_world = transformation(object.com, object.pose.pos, object.pose.orn)
        com_pixel = world_to_pixel(com_world[:2])
        com_xs = np.linspace(com_pixel[0]-(com_marker_width-1)/2,
                                com_pixel[0]+(com_marker_width-1)/2,
                                com_marker_width).astype(np.uint32)
        com_ys = np.linspace(com_pixel[1]-(com_marker_width-1)/2,
                                com_pixel[1]+(com_marker_width-1)/2,
                                com_marker_width).astype(np.uint32)
        if success:
            for com_x in com_xs:
                for com_y in com_ys:
                    try:
                        image[com_y, com_x] = 0.5
                    except:
                        success = False
                        #raise Exception('Object is at the edge of the image! Increase scale and try again.')
        return image, success

    def plot_image(image):
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()


    images = []
    for object in world.objects:
        image, success = get_object_training_image(object)
        #plot_image(image)
        if success:
            images.append(image)
        else:
            print('failed')
            return None, success
    return images, success

def main(args, vis_tower=False):
    # This is a dictionary from stable/unstable label to what subsets of [COG_Stable, PW_Stable] to include.
    difficulty_types = {
        0: [[True, True], [True, False], [False, True], [False, False]],
        1: [[True, True], [True, False], [False, True], [False, False]]
    }

    # specify the number of towers to generate
    num_towers = args.towers_per_cat * 4 * 2
    # specify whether to use a finite set of blocks, or to generate new blocks
    # for each tower
    use_block_set = False
    # the number of blocks in the finite set of blocks
    block_set_size = 1000
    # generate the finite set of blocks
    if use_block_set:
        block_set = [Object.random(f'obj_{i}') for i in range(block_set_size)]
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

                count = 0
                while count < args.towers_per_cat:
                    # print the information about the tower we are about to generate
                    stability_type = "stable" if stable else "unstable"
                    stability_type += "/cog_stable" if cog_stable else "/cog_unstable"
                    stability_type += "/pw_stable" if pw_stable else "/pw_unstable"
                    print(f'{count}/{args.towers_per_cat}\t{stability_type} {num_blocks}-block tower')

                    # generate random blocks. Use the block set if specified. otherwise
                    # generate new blocks from scratch. Save the block names if using blocks
                    # from the block set
                    if use_block_set:
                        blocks = sample_with_replacement(block_set, k=num_blocks)
                    else:
                        blocks = [Object.random(f'obj_{ix}') for ix in range(num_blocks)]

                    # generate a random tower
                    tower = build_tower(blocks, 
                                        stable=stable, 
                                        pairwise_stable=pw_stable, 
                                        cog_stable=cog_stable)
                    
                    if tower is None:
                        continue
                    
                    # NOTE: this has to be done before the sim is run or else
                    # the images will be of the object final positions
                    if args.save_images:
                        w = World(tower)
                        training_images, success = generate_training_images(w)
                        if success:
                            images.append(training_images)
                        else:
                            continue

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
        if num_blocks == 2:
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
    filename = get_filename(num_towers, use_block_set, block_set_size, args.suffix)
    print('Saving to', filename)
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)

def vector_to_block(vec):
    block = Object('', Dimensions(*vec[4:7]), vec[0], vec[1:4], vec[14:17])
    block.pose = Pose(Position(*vec[7:10]), Quaternion(*vec[10:14]))
    return block

def zoom_images():
    filename = 'learning/data/random_blocks_(x10000)_2to5blocks_uniform_density.pkl'
    with open(filename, 'rb') as handle:
        old_dataset = pickle.load(handle)
    dataset = {}
    num_blocks = [2]
    for num_blocks in num_blocks:
        all_vec_blocks = old_dataset[f'{num_blocks}block']['towers']
        labels = old_dataset[f'{num_blocks}block']['labels']
        old_images = old_dataset[f'{num_blocks}block']['images']
        images = []
        
        for i, vec_blocks in enumerate(all_vec_blocks):
            print(str(i)+'/5000')
            
            tower = [vector_to_block(vec_block) for vec_block in vec_blocks]
            w = World(tower)
            training_images, _ = generate_training_images(w)
            images.append(training_images)
        

        data = {
            'towers': all_vec_blocks,
            'labels': labels,
            'images': images
        }

        dataset[f'{num_blocks}block'] = data
        
    # save the generate data
    filename = 'new_2.plk'
    print('Saving to', filename)
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--output', type=str, required=True, help='where to save')
    parser.add_argument('--max-blocks', type=int, required=True)
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--save-images', action='store_true')
    parser.add_argument('--towers-per-cat', default=5000, type=int)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    if args.debug:
        import pdb; pdb.set_trace()

    main(args, vis_tower=False)
    #zoom_images()
