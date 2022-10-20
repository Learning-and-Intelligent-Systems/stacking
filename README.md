# Learning Feasibility

This repo has code for discovering object-level latent spaces based on interactions with that object. This
is an extension of the [stacking](https://github.com/Learning-and-Intelligent-Systems/stacking) repository and
Tmuch of the robot functionality comes from the [rachelholladay/pb_robot](https://github.com/rachelholladay/pb_robot) repository.

## Installation

Dependencies: python3, git

1. Install  `latent_feasibility` 
    1. Clone this repository.
    2. Create a virual environment for this project.
        1. ```cd latent_feasibility```
        2. ```virtualenv .venv --python=/usr/bin/python3```
        3. ```source .venv/bin/activate```
    2. Install dependencies.
        1. ```xargs -n1 pip3 install < requirements.txt```
        2. We are currently using PyTorch `1.9.1`: ```pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html```
2. Install [pb_robot](https://github.com/mike-n-7/pb_robot) using the instructions below, not the ones in the repo's README
    1. Clone `pb_robot` outside of `latent_feasibility`.
    2. Compile the IKFast library for the panda
        1. ```cd pb_robot/src/pb_robot/ikfast/franka_panda```
        2. ```python3 setup.py build``` (for errors, see **troubleshooting** below)           
3. Install [pddlstream](https://github.com/caelan/pddlstream) 
    1. Clone `pddlstream` outside of `latent_feasibility`.
    2. Follow installation instructions there.
4. Create a symlink to required repos (this assumes you cloned `pb_robot` and `pddl_stream` to your home directory). Run these commands from the top directory `latent_feasibility`.
    1. ```ln -s <path-to-pb_robot>/pb_robot/src/pb_robot .```
    2. ```ln -s <path-to-pddstream>/pddlstream/pddlstream .```

## Grasping Dataset

To use the grasping modules (specifically from `pb_robot/planners/antipodalGraspPlanner.py`), you will need to download a dataset
containing URDFs:
1. Download URDFs and object meshes from [here](https://drive.google.com/file/d/1ooOmft1K2lemZXJL62In8sgNzvuzxRWE/view?usp=sharing).
2. Unzip the archive: `unzip shapenet-sim.zip`
3. Set the `SHAPENET_ROOT` environment variable to the location of the dataset: `export SHAPENET_ROOT=<path-to-dataset>/shapenet-sem`


## Run


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



