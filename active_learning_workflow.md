## Active Learning Experiment Workflow
By Izzy Brand, Sebastian Castro, Caris Moses, Michael Noseworthy

## Overview
Machines:
* Panda + control box -- the robot
* RTK -- runs Panda agent, robot interface, and vision
* Desktop machine -- runs planning server and active learning

## RTK Setup
Connecting to the robot

Starting the Franka interface 

```
cd ~/catkin_ws
./franka.sh master
rosrun franka_interface interface.py
```

Interactive Python console to reset the robot position/error state
```
roscd franka_interface/scripts
python3.7 -i interactive.py
>> arm.move_to_neutral()
>> arm.resetErrors()
```

Starting vision server

```
taskset --cpu-list 10,11 roslaunch panda_vision vision.launch
```

Starting the Panda agent server

```
taskset --cpu-list 4,5,8,9 rosrun stacking_ros panda_agent_server.py --num-blocks 10 --real -use-vision
```

## Desktop Setup
Starting the task and motion planning server

```
python stacking_ros/scripts/planning_server.py --blocks-file learning/domains/towers/final_block_set_10.pkl --use-vision --alternate-orientations --num-blocks 10 --max-tries 2 
```

Active learning (starting from scratch)

```
python -m learning.experiments.active_train_towers --exec-mode real --use-panda-server --block-set-fname learning/domains/towers/final_block_set_10.pkl --n-epochs 2 --n-acquire 3 --sampler sequential
```

Active learning (restarting from existing results)

```
python -m learning.experiments.restart_active_train_towers --exp-path learning/experiments/logs/exp-20210218-132207
```

## Video collection laptop setup
FFMpeg magic command
