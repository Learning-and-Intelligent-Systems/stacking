# Execution Guide
This guide will allow you to reproduce several of our experiments, in simulation and in hardware.

## Generating a Block Set
TODO: CSV to block set

---

## Running Tower Stacking
The `run_towers.py` file allows you to run randomly generated towers from a set of blocks. This will help you verify that all the planning and execution tools work, without yet running any of the active learning pipeline.

To run the most basic form of tower stacking in simulation:

```
python3 -m run_towers --blocks-file learning/domains/towers/final_block_set_10.pkl --num-blocks 10
```

To enable placing blocks in their home position, you can use the `--alternate_orientations` flag:

```
python3 -m run_towers --blocks-file learning/domains/towers/final_block_set_10.pkl --num-blocks 10 --alternate-orientations
```

You can also build towers on the real robot and/or by detecting blocks using the RealSense cameras using the `--real` and `--use-vision` flags, respectively:

```
python3 -m run_towers --blocks-file learning/domains/towers/final_block_set_10.pkl --num-blocks 10 --real --use-vision
```

Finally, you can separate planning and execution using the ROS planning server, which is enabled by the `--use-planning-server` flag. In two separate terminals, run the following:

```
python3 -m run_towers --blocks-file learning/domains/towers/final_block_set_10.pkl --num-blocks 10 --use-planning-server

python3 stacking_ros/scripts/planning_server.py --blocks-file learning/domains/towers/final_block_set_10.pkl --num-blocks 10 --alternate-orientations 
```

---

## Running Active Learning
TODO: Run options and restarts

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

---

## Validating a Trained Model
TODO: Plots

Evaluation
```
python3 -m learning.experiments.run_towers_evaluation --real --use-vision --blocks-file learning/domains/towers/eval_block_set_9.pkl --towers-file learning/experiments/logs/robot-seq-init-sim-20210219-131924/evaluation_towers/cumulative-overhang/<model-type>/towers_40.pkl
```

---

## Find COM, solve for stable tower, find and execute plan to build tower
TODO: This is old and was in the original README. Does this still hold, and if so, where should it go?

The following command will find the COM of the given blocks, find the tallest stable tower, and plan and execute the construction of that tower.
```
cd stacking
python3 -m run
```
Arguments:
  - ```--num-blocks (int)```: Number of blocks to use
  - ```--agent ([teleport or panda])```: ```panda``` to have the robot plan and execute actions, ```teleport``` for no agent