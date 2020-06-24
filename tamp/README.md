# TAMP Example

This is an example of doing pick and place operations with the Franka Emika Panda in pybullet. We use [PDDLstream](https://github.com/caelan/pddlstream) as the task and motion planner and use [pb_robot](https://github.com/mike-n-7/pb_robot) as pybullet wrapper. This example is essentially a modified version of an this [example within PDDLstream](https://github.com/caelan/pddlstream/tree/stable/examples/pybullet/kuka).

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
2. Install [pddlstream](https://github.com/caelan/pddlstream) 
    1. follow installation instructions there
3. Install tampExample (this repo)
    1. Clone tampExample
    2. ```cd tampExample/scripts```
    3. ```ln -s ../src tamp```
4. Create a symlink to required repos (this assumes you cloned pb_robot and pddl_stream to your home directory)
    1. ```ln -s ~/pb_robot/src/pb_robot .```
    2. ```ln -s ~/pddlstream/pddlstream .```


### Troubleshooting

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





## Usage

```
cd tampExample/scripts
python -m run_stacking
```

```run_stacking.py``` sets up the problem, calls the planner and then executes the plan (if one was found). The scripts folder also has the two key pddl files. `domain_stacking.pddl` defines the predicates and the actions. `stream_stacking.pddl` defines the streams, which generate values to satify the actions. The various streams are implemented in `src/tampExample/primitives.py`. Within these streams, for this pick and place example, we opt to define grasp sets via TSRs and execute path planning with a bi-directional RRT. These are not significant (truly, they were made out of convenience) and thus could be swapped with any grasp set definition and path planner.  
