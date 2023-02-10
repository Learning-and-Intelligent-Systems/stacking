# Installation
You can choose to locally install this repository and its dependencies, or use our Docker based workflow.

## Local Installation

Dependencies: Ubuntu 18.04, Python 3.6, Git

1. Install [`pb_robot`](https://github.com/mike-n-7/pb_robot) using the instructions below, not the ones in the repo's README
    1. Install pb_robot dependencies
        1. ```pip3 install numpy pybullet recordclass catkin_pkg IPython networkx scipy numpy-quaternion```
        2. ```pip3 install git+https://github.com/mike-n-7/tsr.git```
    2. Clone pb_robot
    3. Compile the IKFast library for the panda
        1. ```cd pb_robot/src/pb_robot/ikfast/franka_panda```
        2. ```python3 setup.py build``` (for errors, see **troubleshooting** below)
2. Install stacking
    1. Clone this repo
    1. Install dependencies
        1. ```cd stacking```
        2. ```xargs -n1 pip3 install < requirements.txt```
        3. NOTE: I got the following errors for the following lines in requirements.txt (although everything afterwards still worked...)
            1. ```-e git://github.com/hauptmech/odio_urdf.git@1f9ecb4e7833957c11cc707fa4c1b781ceb70ae8#egg=odio_urdf```-->```-e option requires 1 argument```
            2. ```tsr @ git+https://github.com/mike-n-7/tsr.git@aa92079b215fddf58075b71fc07a0ddb17e309af```-->```ERROR: Invalid requirement: '@'```
            
3. Install [`pddlstream`](https://github.com/caelan/pddlstream) 
    1. follow installation instructions there
4. Create a symlink to required repos (this assumes you cloned pb_robot and pddl_stream to your home directory). Run these commands from the top directory of this repo.
    1. ```ln -s ~/pb_robot/src/pb_robot .```
    2. ```ln -s ~/pddlstream/pddlstream .```

### ROS Installation
If you plan to interface with the robot and/or use the multi-node planning and execution mode, you will additionally need to set up Robot Operating System (ROS). 

Speficially, we use ROS Melodic. For more detailed instructions, look at the [`stacking_ros` README](../stacking_ros/README.md).

---

## Docker Installation

To run Docker containers with graphics (Gazebo, RViz, GPU support, etc.), first install the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker).

We recommend creating a top-level folder and cloning this repo inside of it. For example:

```
mkdir my_project
cd my_project
git clone https://github.com/Learning-and-Intelligent-Systems/stacking.git
cd stacking
```

To build the Docker image, run 

``` 
make build
```

To get access to display capabilities (like PyBullet, etc.), you will need to run this once every time you log in to your machine:

```
make xhost-activate
```

Once the image is built, you can start the container with

```
make term
```

Note that this Docker image will mount the Catkin generated build files to `build`, `devel`, and `logs` folders on your host so you don't need to rebuild the Catkin workspace every time and you can modify the Python code from your host machine. If at some point you need to do a clean rebuild, ensure to remove these folders.

```
cd ..
sudo rm -r build/ devel/ logs/
```

Refer to the [`Makefile`](../Makefile) and [`Dockerfile`](../Dockerfile) for more information or to modify things as necessary.

## Troubleshooting

Update: This repo will now work by default with Python3.7. This troubleshooting may still be useful for setting up the repo using a different Python version. 

On macOS Catalina using a Python3.7 virtualenv, building pb_robot with `python setup.py build` failed with the following error

```./ikfast.h:41:10: fatal error: 'python3.6/Python.h' file not found```

The compiler can't find the appropriate python header. The solution is to first locate the header:

```
$ find /usr/local/Cellar/ -name Python.h
/usr/local/Cellar//python/3.7.7/Frameworks/Python.framework/Versions/3.7/include/python3.7m/Python.h
/usr/local/Cellar//python@3.8/3.8.2/Frameworks/Python.framework/Versions/3.8/include/python3.8/Python.h
```

which prints the python include directories. I wanted to use 3.7, so then I set the environment variable

```export CPLUS_INCLUDE_PATH=/usr/local/Cellar//python/3.7.7/Frameworks/Python.framework/Versions/3.7/include/```

and finally modify `pb_robot/src/pb_robot/ikfast/ikfast.h` by changing

```
#include "python3.6/Python.h" -> #include "python3.7m/Python.h"
```