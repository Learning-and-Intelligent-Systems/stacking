import argparse
import pickle
import numpy as np
import os

from PIL import Image, ImageOps, ImageDraw, ImageFont


def concat_images(image_paths, size, shape=None, txs=[]):
    # Open images and resize them
    vpadding = 15
    hpadding = 5
    width, height = size

    
    valid_image_paths = [p for p in image_paths if os.path.exists(p)]
    valid_images = map(Image.open, valid_image_paths)
    valid_images = [ImageOps.fit(image, size, Image.ANTIALIAS) 
              for image in valid_images]

    img_ix = 0
    images = []
    for p in image_paths:
        if os.path.exists(p):
            images.append(valid_images[img_ix])
            img_ix += 1
        else:
            images.append(None)
    
    # Create canvas for the final image with total size
    shape = shape if shape else (1, len(images))
    image_size = (width * shape[1] + 9*hpadding, height * shape[0] + 2*vpadding)
    image = Image.new('RGB', image_size)
    
    draw = ImageDraw.Draw(image)
    #font = ImageFont.truetype("sans-serif", 16)
    # Paste images into final image
    for row in range(shape[0]):
        for col in range(shape[1]):
            offset = width * col+ col*hpadding, height * row + row*vpadding
            idx = row * shape[1] + col
            if images[idx] is None:
                continue
            image.paste(images[idx], offset)
    # for col in range(shape[1]):
    #     draw.text((width*col + width/2, 10),f"tx={txs[col]}",(0,0,0))
    return image


if __name__ == '__main__':
    
    image_ids = range(0, 10)
    
    # Get list of image paths
    task = 'min_contact'
    folder = f'images/{task}'

    models = ['simple-model', 'noisy-model', 'learned']
    #tasks = ['tallest', 'min-contact', 'cumulative-overhang']
    image_paths = []
    for ix, mname in enumerate(models):
        for tx in image_ids:
            image_paths.append(f'{folder}/{mname}/{tx}.png')
    if task == 'tallest':
        image = concat_images(image_paths, (175, 275), (len(models), len(image_ids)))
    else:
        image = concat_images(image_paths, (175, 250), (len(models), len(image_ids)))
    image.save(f'{folder}/comparison.png')