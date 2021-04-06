#/usr/bin/sh

# Run this script from the al_videos directory. 
# Make sure all original full length videos are in the original/ folder.

# First, make a 128x version of the video if it doesn't exist yet.
cd eval_videos 
for i in *.mp4; do
    echo $i
    if [ ! -f ../eval_spedup/$i ]; then
        ffmpeg -i $i -filter:v "setpts=0.125*PTS" ../eval_spedup/$i 
    fi
done
