# stacking_ros

This package provides a Task and Motion Planning (TAMP) server that offers ROS services to interact with the Panda stacking agent.

## Setup

First, ensure this package is in a Catkin workspace (e.g. `/catkin_ws/src/stacking/stacking_ros`).

Now, build the Catkin workspace

```
catkin_build -DCPYTHON_EXECUTABLE=$(which python3.7)
```

Ensure that this Catkin workspace is being sourced:

```
setup /catkin_ws/devel/setup.bash
```

To check that the package works correctly, try running some of these commands:

```
rospack find stacking_ros
rosmsg list | grep stacking
```

---

## Tower Simulation Usage
The `scripts/planning_server.py` file holds the service server. To run this, you have a few options.

### rosrun
```
rosrun stacking_ros planning_server.py --num-blocks 4
```

To do this, you will need to ensure the `stacking` repo is on the Python path. You can force this, e.g.
```
export PYTHONPATH=$PYTHONPATH:/catkin_ws/src/stacking
```

### Python
First, go to the top-level folder of the `stacking` repository. Then,

```
python3.7 stacking_ros/scripts/planning_server.py --blocks-file learning/domains/towers/final_block_set.pkl --num-blocks 10

```

---

## Active Learning Usage
First, start the planning server with the block list of choice. Keep it handy because we need to use it for every separate call below.

```
rosrun stacking_ros planning_server.py --blocks-file learning/domains/towers/final_block_set.pkl --num-blocks 10
```

Then, start the Panda agent server with the same block list.

```
rosrun stacking_ros panda_agent_server.py --blocks-file learning/domains/towers/final_block_set.pkl --num-blocks 10
```

Finally, start active learning.

```
python3.7 -m learning.experiments.active_train_towers --exec-mode sim --use-panda-server --block-set-fname learning/domains/towers/final_block_set.pkl
```
