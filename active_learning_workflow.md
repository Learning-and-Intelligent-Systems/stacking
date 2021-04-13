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
roslaunch franka_interface interface.launch
```

Interactive Python console to reset the robot position/error state
```
roscd franka_interface/scripts
python3 -i interactive.py
>> arm.move_to_neutral()
>> arm.resetErrors()
```

Starting vision server

```
taskset --cpu-list 10,11 roslaunch panda_vision vision.launch
```

Starting the Panda agent server

```
taskset --cpu-list 4,5,8,9 rosrun stacking_ros panda_agent_server.py --num-blocks 10 --real --use-vision --use-planning-server
```

## Desktop Setup
Starting the task and motion planning server

```
python3 stacking_ros/scripts/planning_server.py --blocks-file learning/domains/towers/final_block_set_10.pkl --use-vision --alternate-orientations --num-blocks 10 --max-tries 2 
```

Active learning (starting from scratch)

```
python3 -m learning.experiments.active_train_towers --exec-mode real --use-panda-server --block-set-fname learning/domains/towers/final_block_set_10.pkl --n-epochs 20 --n-acquire 10 --sampler sequential --exp-name robot-seq-init-sim --n-samples 100000
```

Active learning (restarting from existing results)

```
python3 -m learning.experiments.restart_active_train_towers --exp-path learning/experiments/logs/exp-20210218-161131
```

To evaluate current progress, run this with the appropriate arguments (tx).
```
python3 -m learning.evaluate.plot_model_accuracy --exp-paths learning/experiments/logs/robot-seq-init-sim-20210219-131924 --max-acquisitions <tx> --plot-step 1 --test-set-fname learning/evaluate/test_datasets/eval_blocks_test_dataset.pkl --output-fname test
```

If a tower is mislabeled during acquisition step ```<tx>```, run the following AFTER all 
towers for step ```<tx>``` have been labeled, and ```acquired_<tx>.pkl``` has been generated.
```<tn>``` corresponds to the tower number integer in the towers file name (0 indexed).
 ```<l>``` is the correct label.
If any files from step ```<tx>+1``` have  been generated then delete them. (They will be regenerated/trained on restart)
```
python3 -m learning.experiments.fix_tower_label --exp-path <path> --acquisition-step <tx>
--tower-number <tn> --label <l>
```

## Video collection laptop setup
FFMpeg magic command

## Evaluation Workflow

1. Start franka interface on RTK machine
2. Start vision on RTK machine 
3. Start planning server on GPU machine (use the learning/domains/towers/eval_block_set_9.pkl)
4. Run the following to start evaluating where ```model-type``` in [learned, noisy-model, simple-model]
```
python3 -m learning.experiments.run_towers_evaluation --real --use-vision --blocks-file learning/domains/towers/eval_block_set_9.pkl --towers-file learning/experiments/logs/robot-seq-init-sim-20210219-131924/evaluation_towers/cumulative-overhang/<model-type>/towers_40.pkl
```
