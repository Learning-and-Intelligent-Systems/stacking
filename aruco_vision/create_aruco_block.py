""" Massachusetts Institute of Technology

Izzybrand, 2020
"""
import argparse
import cv2
import numpy as np
import pickle
from PIL import Image
from rotation_util import *

# Load the predefined dictionary
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

ppcm = 300 # pixels per cm specifies the resolution of the saved image
marker_scale = 0.9 # scale of the aruco tag on the face
dpi = ppcm*2.54

def generate_texture(block_id, block_dimensions):
    info = {'block_id': block_id, 'dimensions': np.array(block_dimensions)}
    d_x, d_y, d_z = block_dimensions
    face_dimensions_list = np.array([[d_y, d_z],
                                     [d_y, d_z],
                                     [d_x, d_z],
                                     [d_x, d_z],
                                     [d_x, d_y],
                                     [d_x, d_y]])
    for i in range(6):
        # get the dimensions of the face
        face_dimensions_cm = face_dimensions_list[i] * 100
        # the marker corresponding to this face
        marker_id = block_id*6 + i

        # Generate the marker
        marker_size_cm = face_dimensions_cm.min() * marker_scale
        marker_size = int(marker_size_cm * ppcm)
        marker_image = np.zeros((marker_size, marker_size), dtype=np.uint8)
        marker_image = cv2.aruco.drawMarker(dictionary, marker_id, marker_size, marker_image, 1)

        # and insert the marker into an image for the face
        face_dimensions_px = (face_dimensions_cm*ppcm).astype(int)
        face_image = np.ones(face_dimensions_px, dtype=np.uint8) * 255
        ul_corner_idx = (face_dimensions_px - marker_size) // 2
        br_corner_idx = (face_dimensions_px - marker_size) // 2 + marker_size
        face_image[ul_corner_idx[0]:br_corner_idx[0], ul_corner_idx[1]:br_corner_idx[1]] = marker_image

        # draw the block_id on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        # org = (10, 10)
        font_scale = 2
        thickness = 2
        text = str(block_id)
        (x,y), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        org = (10,10+y)
        face_image = cv2.putText(face_image, text, org, font,
                   font_scale, 0, thickness, cv2.LINE_AA, bottomLeftOrigin=False)

        # add a 2-pixel border to the image to show where to cut with scissors
        face_image[[0,1,-2,-1]] = 0
        face_image[:,[0,1,-2,-1]] = 0

        # save the image at the correct size (PPCM) to print
        pil_face_image = Image.fromarray(face_image)
        pil_face_image.save(f'tags/block_{block_id}_tag_{marker_id}.png', format='PNG', dpi=(dpi, dpi))

        # add the marker size to the info dict
        info[marker_id] = { 'marker_size_cm': marker_size_cm }

    return info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create aruco tags for a block.')
    parser.add_argument('block_id', type=int, help='The id number of this block')
    parser.add_argument('--dimensions', nargs=3, metavar=('dx', 'dy', 'dz'), type=float,
                    help='Dimensions of the block in meters', default=None)
    args = parser.parse_args()

    info = generate_texture(args.block_id, args.dimensions)
    info_filename = f'tags/block_{args.block_id}_info.pkl'
    with open(info_filename, 'wb') as f:
        pickle.dump(info, f)

    print(f'Saved block info to {info_filename}')
