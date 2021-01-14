# Installing librealsense

## On Linux

See the [librealsense github](https://github.com/IntelRealSense/librealsense)

## On Mac

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
