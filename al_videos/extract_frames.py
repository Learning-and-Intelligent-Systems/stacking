import argparse
import cv2
import os

def video_to_frame(video_filename, frame_id):
    """Extract frames from video"""
    cap = cv2.VideoCapture(video_filename)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    if cap.isOpened() and video_length > 0:
        count = 0

        cap.set(cv2.CAP_PROP_POS_MSEC, frame_id*1000.)
        success, image = cap.read()
        return image
           

if __name__ == '__main__':
    """
    Usage: 
    (1) Create a directory where you want all the images to be saved.
    (2) Within that directory, make a file called timestamps.txt
    (3) On each line in that file, write <video_fname>, <video_timestamp_in_seconds> for the frame the tower you want is in.
        - Note this should be relative from the al_videos directory.
    (4) Run this script with the argument --dir pointing towards this folder.
        - The pictures will be saved to this folder.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    args = parser.parse_args()

    with open(f'{args.dir}/timestamps.txt', 'r') as handle:
        for lx, line in enumerate(handle.readlines()):
            if len(line) < 2: continue
            fname, ix = line.strip().split(',')
            print(fname, int(ix))
            frame = video_to_frame(fname, int(ix))
            #cv2.imshow('img', frame)
            crop = frame[250:450, 200:350]
            cv2.imwrite(os.path.join(args.dir, f'{lx}.png'), crop)
            cv2.waitKey(0)
