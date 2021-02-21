#/usr/bin/sh

# Run this script from the al_videos directory. 
# Make sure all original full length videos are in the original/ folder.

# First, make a 128x version of the video if it doesn't exist yet.
cd original 
for i in *.mp4; do
    echo $i
    if [ ! -f ../spedup/$i ]; then
        ffmpeg -i $i -filter:v "setpts=0.0078125*PTS" ../spedup/$i 
    fi
done

# Create a list of all the video names that will be concatenated with each other.
cd ../spedup
if [ -f ../video_list.txt ]; then
    rm ../video_list.txt
fi

touch ../video_list.txt
for i in *.mp4; do
    echo "file 'spedup/$i'" >> ../video_list.txt
done

# Concatenate the sped up videos.
cd ../
ffmpeg -f concat  -i video_list.txt -c copy training.mp4