import cv2
import json
import numpy as np
from PIL import Image
from rotation_util import *

# Load the predefined dictionary
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

ppcm = 300 # pixels per cm
marker_scale = 0.9 # scale of the aruco tag on the face

dpi = ppcm*2.54


def generate_texture(block_id, block_dimensions):
    images = []
    info = []
    d_x, d_y, d_z = block_dimensions
    face_dimensions_list = np.array([[d_y, d_z],
                                     [d_y, d_z],
                                     [d_x, d_z],
                                     [d_x, d_z],
                                     [d_x, d_y],
                                     [d_x, d_y]])
    face_tranlations_list = [[ d_x/2., 0, 0],
                             [-d_x/2., 0, 0],
                             [0,  d_y/2., 0],
                             [0, -d_y/2., 0],
                             [0, 0,  d_z/2.],
                             [0, 0, -d_x/2.]]
    face_rotations_list = [eul_to_rot([0, 0, 0]),
                           eul_to_rot([0, 0, np.pi]),
                           eul_to_rot([0, np.pi, 0]),
                           eul_to_rot([0, -np.pi, 0]),
                           eul_to_rot([0, 0, np.pi]),
                           eul_to_rot([0, 0, -np.pi])]
    face_names = ['right', 'left', 'top', 'bottom', 'front', 'back']
    for i in range(6):
        # get the dimensions of the face
        face_dimensions = face_dimensions_list[i]

        # the marker corresponding to this face
        marker_id = block_id*6 + i

        # Generate the marker
        marker_size_cm = face_dimensions.min() * marker_scale
        marker_size = int(marker_size_cm * ppcm)
        marker_image = np.zeros((marker_size, marker_size), dtype=np.uint8)
        marker_image = cv2.aruco.drawMarker(dictionary, marker_id, marker_size, marker_image, 1)

        # and insert the marker into an image for the side
        side_image = np.ones(face_dimensions*ppcm, dtype=np.uint8) * 255
        ul_corner_idx = (face_dimensions*ppcm - marker_size) // 2
        br_corner_idx = (face_dimensions*ppcm - marker_size) // 2 + marker_size
        side_image[ul_corner_idx[0]:br_corner_idx[0], ul_corner_idx[1]:br_corner_idx[1]] = marker_image

        # add edges
        side_image[[0,-1]] = 0
        side_image[:,[0,-1]] = 0

        # cv2.imwrite(f'tags/block_{block_id}_face_{face}.png', side_image)
        pil_side_image = Image.fromarray(side_image)
        pil_side_image.save(f'tags/block_{block_id}_face_{i}.png', format='PNG', dpi=(dpi, dpi))
        # cv2.imshow('display', side_image)
        # cv2.waitKey(500)

        images.append(side_image)
        face_info = {
                    'name': face_names[i],
                    'marker_id': marker_id,
                    'marker_size_cm': marker_size_cm,
                    'translation_cog_to_marker': face_tranlations_list[i],
                    'rotation_cog_to_marker': face_rotations_list[i]
                    }

        info.append(face_info)

    return images, info


if __name__ == '__main__':
    block_dimensions = np.array([6,10,6])
    block_id = 4
    images, info = generate_texture(block_id, block_dimensions)
    with open(f'tags/block_{block_id}_info.json', 'w') as f:
        json.dump(info, f)

