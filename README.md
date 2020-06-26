# stacking

This repo has code for using a simulation-based particle filter to estimate the center of mass of blocks, then 
solving for the tallest stable tower (in expectation). Finally, the tower is constructed using a panda robot in simulation.
The robot functionality comes from [rachelholladay/pb_robot](https://github.com/rachelholladay/pb_robot) repository.

The particle filter and tower solver run in Python3 and the robot runs in Python2.

## Installation

This should be run with Python3.7, NOT Python2

1. Install [pb_robot](https://github.com/mike-n-7/pb_robot) using the instructions below, not the ones in the repo's README
    1. Install pb_robot dependencies
        1. ```pip3 install numpy pybullet recordclass catkin_pkg IPython networkx scipy```
        2. ```pip3 install git+https://github.com/mike-n-7/tsr.git```
    2. Clone pb_robot
    3. Compile the IKFast library for the panda
        1. ```cd pb_robot/src/pb_robot/ikfast/franka_panda```
        2. ```python setup.py build``` (for errors, see **troubleshooting** below)
2. Install stacking
    1. Install dependencies
        1. ```xargs -n1 pip3 install < requirements.txt```
3. Install [pddlstream](https://github.com/caelan/pddlstream) 
    1. follow installation instructions there
4. Create a symlink to required repos (this assumes you cloned pb_robot and pddl_stream to your home directory). Run these commands from the top directory of this repo.
    1. ```ln -s ~/pb_robot/src/pb_robot .```
    2. ```ln -s ~/pddlstream/pddlstream .```
  
## Run Particle Filter and Tallest Stable Tower Solver

```
cd stacking
python3 -m run --save-tower
```

The ```--save-tower``` argument is optional. Use it if you would like the robot to later construct the found tower. The location 
of the directory where the tower files are saved will print out to the screen

## Have a Panda Construct the Tower (This has been extended to using PDDLStream to build towers - see below)

```
cd stacking
python2 -m build_tower --tower-dir TOWER_DIR
```

The ```--tower-dir``` is required, and ```TOWER_DIR``` is the directory output from the previous step (if the --save-tower argument was used).

## PDDLStream Integration

Another way to have the robot build a tower is by using the PDDLStream planner once a tower configuration has been output. 

```
python -m tamp.run_stacking
```

`run_stacking.py` sets up the problem, calls the planner and then executes the plan (if one was found). The tamp folder also has the two key pddl files. `domain_stacking.pddl` defines the predicates and the actions. `stream_stacking.pddl` defines the streams, which generate values to satify the actions. The various streams are implemented in `tamp/primitives.py`. Within these streams, for this pick and place example, we opt to define grasp sets via TSRs and execute path planning with a bi-directional RRT. These are not significant (truly, they were made out of convenience) and thus could be swapped with any grasp set definition and path planner.  

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



