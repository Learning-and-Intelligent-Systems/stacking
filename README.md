# stacking

This repo has code for using a simulation-based particle filter to estimate the center of mass of blocks, then 
solving for the tallest stable tower (in expectation). Finally, the tower is constructed using a panda robot in simulation.
The robot functionality comes from [rachelholladay/pb_robot](https://github.com/rachelholladay/pb_robot) repository.

The particle filter and tower solver run in Python3 and the robot runs in Python2.

## Installation

1. Install pb_robot
    1. Install pb_robot dependencies
        1. ```pip2 install numpy pybullet recordclass catkin_pkg IPython networkx```
        2. ```pip2 install git+https://github.com/personalrobotics/tsr.git```
    2. Clone pb_robot from [here](https://github.com/rachelholladay/pb_robot)
    3. Comment out line 6 of pb_robot/src/pb_robot/\__init__.py (```import vobj```) 
    4. Compile the IKFast library for the panda
        1. ```cd pb_robot/src/pb_robot/ikfast/franka_panda```
        2. ```python2 setup.py build```
2. Install stacking
    1. Install dependencies
        1. ```xargs -n1 pip3 install < requirements.txt```
    2. Clone this repo
3. Create a symlink to pb_robot in the stacking repo (this assumes you cloned stacking and pb_robot to your home directory)
    1. ```ln -s ~/pb_robot/src/pb_robot ~/stacking```
  
## Run Particle Filter and Tallest Stable Tower Solver

```
cd stacking
python3 -m run --save-tower
```

The ```--save-tower``` argument is optional. Use it if you would like the robot to later construct the found tower. The location 
of the directory where the tower files are saved will print out to the screen

## Have a Panda Construct the Tower

```
cd stacking
python2 -m build_tower --tower-dir TOWER_DIR
```

The ```--tower-dir``` is required, and ```TOWER_DIR``` is the directory output from the previous step (if the --save-tower argument was used).

## PDDLStream Integration

Another way to have the robot build a tower is by using the PDDLStream planner once a tower configuration has been output. Please follow the README in the `tamp` folder. PDDLStream uses Python3.
