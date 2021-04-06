# Execution Guide
This guide will allow you to reproduce several of our experiments, in simulation and in hardware.

## Generating a Block Set
TODO: CSV to block set

---

## Running Tower Stacking
TODO: Run options (sim vs. hardware, use vision, ROS planning server, etc.)

```
python3 -m run_towers --blocks-file learning/domains/towers/final_block_set.pkl --num-blocks 10
```

---

## Running Active Learning
TODO: Run options and restarts

```
python3 -m learning.experiments.active_train_towers --exec-mode sim --block-set-fname learning/domains/towers/final_block_set.pkl
```

---

## Validating a Trained Model
TODO: Plots and evaluation

---

## Find COM, solve for stable tower, find and execute plan to build tower
The following command will find the COM of the given blocks, find the tallest stable tower, and plan and execute the construction of that tower.
```
cd stacking
python3 -m run
```
Arguments:
  - ```--num-blocks (int)```: Number of blocks to use
  - ```--agent ([teleport or panda])```: ```panda``` to have the robot plan and execute actions, ```teleport``` for no agent

