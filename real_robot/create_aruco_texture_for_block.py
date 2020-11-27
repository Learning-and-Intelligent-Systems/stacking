import cv2
import numpy as np

# Load the predefined dictionary
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

ppcm = 300 # pixels per cm
marker_scale = 0.9 # scale of the aruco tag on the face



def generate_texture(block_id, block_dimensions):
    images = []
    marker_ids = []
    marker_sizes = []
    for face in range(6):
        # get the dimensions of the face
        dimension_indices = np.array([face, face+1]) % 3
        face_dimensions = block_dimensions[dimension_indices]

        # the marker corresponding to this face
        marker_id = block_id*6 + face

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

        cv2.imshow('test', side_image)
        cv2.waitKey(300)

        images.append(side_image)
        marker_ids.append(marker_id)
        marker_sizes.append(marker_size_cm)

    return images, marker_ids, marker_sizes


if __name__ == '__main__':
    block_dimensions = np.array([5,6,7])
    block_id = 4
    images, marker_ids, marker_sizes = generate_texture(block_id, block_dimensions)