import argparse
import pickle
import numpy as np
import os

from learning.active.utils import ActiveExperimentLogger
from block_utils import Object
from learning.evaluate.active_evaluate_towers import tallest_tower_regret_evaluation, \
        longest_overhang_regret_evaluation, min_contact_regret_evaluation

from PIL import Image, ImageOps, ImageDraw, ImageFont


def concat_images(image_paths, size, shape=None, txs=[]):
    # Open images and resize them
    width, height = size
    images = map(Image.open, image_paths)
    images = [ImageOps.fit(image, size, Image.ANTIALIAS) 
              for image in images]
    
    # Create canvas for the final image with total size
    shape = shape if shape else (1, len(images))
    image_size = (width * shape[1], height * shape[0])
    image = Image.new('RGB', image_size)
    
    draw = ImageDraw.Draw(image)
    #font = ImageFont.truetype("sans-serif", 16)
    # Paste images into final image
    for row in range(shape[0]):
        for col in range(shape[1]):
            offset = width * col, height * row
            idx = row * shape[1] + col
            image.paste(images[idx], offset)
    for col in range(shape[1]):
        draw.text((width*col + width/2, 10),f"tx={txs[col]}",(0,0,0))
    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', 
                        choices=['tallest', 'overhang', 'min-contact', 'deconstruct'], 
                        default='tallest',
                        help='planning problem/task to plan for')
    parser.add_argument('--block-set-fname', 
                        type=str, 
                        required=True,
                        help='path to the block set file. if not set, args.n_blocks random blocks generated.')
    parser.add_argument('--exp-path', 
                        type=str, 
                        required=True)
    parser.add_argument('--debug',
                        action='store_true',
                        help='set to run in debug mode')
    parser.add_argument('--xy-noise',
                        type=float,
                        required=True,
                        help='noise to add to xy position of blocks')
    
    args = parser.parse_args()
    args.discrete = False
    args.tower_sizes = [5]
    args.max_acquisitions = None
    args.n_towers = 1
    acquisition_steps = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    
    with open(args.block_set_fname, 'rb') as f:
        block_set = pickle.load(f)[:5]

    logger = ActiveExperimentLogger(args.exp_path)
    
    for tx in acquisition_steps:
        args.acquisition_step = tx
        if not os.path.exists(logger.get_figure_path(f'height_{tx}_0.png')):
            tallest_tower_regret_evaluation(logger, block_set, '', args, save_imgs=True)
        if not os.path.exists(logger.get_figure_path(f'overhang_{tx}_0.png')):
            longest_overhang_regret_evaluation(logger, block_set, '', args, save_imgs=True)
        if not os.path.exists(logger.get_figure_path(f'contact_{tx}_0.png')):
            min_contact_regret_evaluation(logger, block_set, '', args, save_imgs=True)
    
    # Get list of image paths
    folder = 'learning/experiments/logs/robot-seq-init-sim-20210219-131924/figures'

    tasks = ['height', 'contact', 'overhang']
    image_paths = []
    for ix, task in enumerate(tasks):
        for tx in acquisition_steps:
            image_paths.append(f'{folder}/{task}_{tx}_0.png')

    image = concat_images(image_paths, (100, 190), (len(tasks), len(acquisition_steps)), acquisition_steps)
    image.save(f'{folder}/qualitative.png')