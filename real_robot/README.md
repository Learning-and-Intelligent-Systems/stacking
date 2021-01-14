# Aruco Vision Pipeline

We will use OpenCVs Aruco tags to localize the blocks with the Intel RealSenseD435 RGBD camera. All six sides of each object have a unique aruco tag.

#### Requirements

 * librealsense  (see below for install tips)
 * Python OpenCV
 * PIL

## Creating Tags for an object

Each object has six unique tags; this script generates those tags according to the given `block_id` and saves them in the `tags` folder. Each tag is saved as a PNG image with a black border. When printed on a regular printer at the default scale (set by the script), you can cut out the tag along the border and it will match the dimensions of each face on the block.

Additionally, this script generates a block info file saved at `tags/block_[num]_info.pkl`. This stores things the size of the aruco tags on that block, and the dimensions of the block which are needed for calibration and pose estimation.

Run the script by

`python create_aruco_block.py [block_id] [--dimensions dx dy dz]`

Print out the resulting tags, cut along the black borders, and apply the block with tape or glue. It does not matter which face the tag is applied to as long as the shape matches.

## Calibrating tags for an object

`python calibrate_aruco_block.py`

## Localizing calibrated objects

`python aruco_block_pose_est.py`

## Installing librealsense

#### On Linux

See the [librealsense github](https://github.com/IntelRealSense/librealsense)

#### On Mac

I found [this page](https://github.com/IntelRealSense/librealsense/issues/5275) to be useful

This install was for getting the realsense working in a virtualenv. I previously installed opencv to a virtualenv following [this article](https://www.pyimagesearch.com/2018/08/17/install-opencv-4-on-macos/), which is where I learned that the `*.so` files can be put in `<virtualenvdir>/lib/python3.7/site-packages`. The exact commands that I ran were:

```
cd ~
git clone https://github.com/IntelRealSense/librealsense
cd librealsense
mkdir build
cd build
cmake ../ -DBUILD_PYTHON_BINDINGS=bool:true
make -j4  # this took a while
sudo make install
cd wrappers/python
cp *.so <virtualenvdir>/lib/python3.7/site-packages/.
```
