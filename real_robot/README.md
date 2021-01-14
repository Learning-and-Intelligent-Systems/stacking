# Aruco Vision Pipeline

We will use OpenCVs Aruco tags to localize the blocks with the Intel RealSenseD435 RGBD camera. All six sides of each object have a unique aruco tag.

## Creating Tags for an object

`python create_aruco_block.py`

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
