# Repository Overview

In addition to this main `stacking` repository, we rely on several other repositories to provide supporting functionality.

* The PyBullet simulation functionality comes from the [`pb_robot`](https://github.com/mike-n-7/pb_robot) repository.
* Task and Motion Planning (TAMP) is enabled by the [`pddlstream`](https://github.com/caelan/pddlstream) repository.
* Interfaces to the robot are enabled by the [`franka_ros_interface`](https://github.com/rachelholladay/franka_ros_interface) repository.
* Block detection with the Intel RealSense cameras is enabled by the [`panda_vision`](https://github.com/carismoses/panda_vision) repository.

---

TODO: Add more specifics on concepts and folder structure

## Task and Motion Planning

The `tamp` folder has the two key pddl files. `domain_stacking.pddl` defines the predicates and the actions. `stream_stacking.pddl` defines the streams, which generate values to satify the actions. The various streams are implemented in `tamp/primitives.py`. Within these streams, for this pick and place example, we opt to define grasp sets via TSRs and execute path planning with a bi-directional RRT. These are not significant (truly, they were made out of convenience) and thus could be swapped with any grasp set definition and path planner.  

## Planning and Learning ROS Servers

The `stacking_ros` folder contains a ROS package with utilities to help split up active learning, task and motion planning, and execution into separate software nodes. These could be run on different terminals or even across different machines provided they are on the same network.

Refer to [the `stacking_ros` README](../stacking_ros/README.md) for more information.