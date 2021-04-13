# Execution Guide
This guide will allow you to reproduce several of our experiments, in simulation and in hardware.

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
Our active learning pipeline has several options, most importantly the models and sampling strategies.

Models include:
* Execution models: Simple simulations with/without noise, PyBullet simulation, or the real robot.
* Action Plan Feasibility (APF) prediction neural network architectures

Sampling strategies include:
* Tower sampling strategy: Sampling randomly, with the BALD objective to maximize information gain, sampling at the full tower vs. subtower level, etc.
* Unlabeled pool sampling strategy:

Refer to [`active_train_towers.py`](../learning/experiments/active_train_towers.py) for more information on these input arguments.

### Experimental Data Folder Structure
When you run an active learning experiment, data is saved in the `learning/experiments/logs` folder. This includes,

* `acquisition_data` : Data for an entire acquisition step (of N towers)
* `towers` : Data for all towers built over all acquisition steps
* `models` : APF prediction model ensembles, with weights at each acquisition steps
* `datasets` : TODO
* `val_datasets` : TODO
* `figures` : Contains validation and/or analysis plots generated for this experiment by other auxiliary scripts
* `args.pkl` : Set of arguments used for restarting training
* `block_placement_data.pkl` : TODO

### Active Learning in Simulation
The following command will run active learning on a noisy simulation model with 2 mm noise.

```
python3 -m learning.experiments.active_train_towers --exec-mode noisy-model --block-set-fname learning/domains/towers/final_block_set_10.pkl --n-epochs 20 --n-acquire 10 --sampler sequential --xy-noise 0.002 --n-samples 100000 --exp-name noisy-sim 
```

By specifying `--exp-name noisy-sim`, this will create a folder `learning/experiments/logs/noisy-sim-<date>` containing all the necessary experimental data.

To restart the same experiment from its last collected label, using the same arguments as originally specified:

```
python3 -m learning.experiments.restart_active_train_towers --exp-path learning/experiments/logs/noisy-sim-<date>
```

### Active Learning on the Panda
Active learning using a simulated or real Panda additionally requires task and motion planning. This is done by setting the `--exec-mode` flag to `sim` or `real`, respectively.

You can choose to run the active learning and the Panda agent in the same process, or splitting active learning and the Panda agent into two software nodes. This is toggled using the `--use-panda-server` flag.

Running all in the same process will look as follows:

```
python3 -m learning.experiments.active_train_towers --block-set-fname learning/domains/towers/final_block_set_10.pkl --n-epochs 20 --n-acquire 10 --sampler sequential --n-samples 100000 --exec-mode sim --exp-name sim-panda
```

Separating active learning and task and motion planning will look as follows. Note that with this option you have to start a Panda agent in another Terminal, and there you can choose whether to additionally split up planning and execution using `--use-planning-server`

```
python3 -m learning.experiments.active_train_towers --block-set-fname learning/domains/towers/final_block_set_10.pkl --n-epochs 20 --n-acquire 10 --sampler sequential --n-samples 100000 --exec-mode sim --exp-name sim-panda-distributed --use-panda-server

rosrun stacking_ros panda_agent_server.py --num-blocks 10 --use-planning-server
```

You can similarly restart existing experiments when using the Panda robot. In fact, you will have to do so more often because of real planning and execution failures.

To restart the same experiment from its last collected label, using the same arguments as originally specified:

```
python3 -m learning.experiments.restart_active_train_towers --exp-path learning/experiments/logs/<experiment-name>
```

---

## Validating a Trained Model
To evaluate current training progress, run this with the appropriate arguments.

```
python3 -m learning.evaluate.plot_model_accuracy --exp-paths learning/experiments/logs/<experiment-name> --max-acquisitions <tx> --plot-step 1 --test-set-fname learning/evaluate/test_datasets/eval_blocks_test_dataset.pkl --output-fname test
```

If a tower is mislabeled during acquisition step ```<tx>```, run the following AFTER all towers for step ```<tx>``` have been labeled, and ```acquired_<tx>.pkl``` has been generated.
```<tn>``` corresponds to the tower number integer in the towers file name (0 indexed).
 ```<l>``` is the correct label.
If any files from step ```<tx>+1``` have  been generated then delete them. (They will be regenerated/trained on restart)
```
python3 -m learning.experiments.fix_tower_label --exp-path <path> --acquisition-step <tx>
--tower-number <tn> --label <l>
```

---

## Evaluating a Trained Model
Finally, you can run a trained model on the simulated or real Panda robot using the `run_towers_evaluation.py` script.

To generate a set of evaluation towers
```
TODO
```

To execute the model on a set of towers:
```
python3 -m learning.experiments.run_towers_evaluation --real --use-vision --blocks-file learning/domains/towers/eval_block_set_9.pkl --towers-file learning/experiments/logs/<experiment-name>/evaluation_towers/cumulative-overhang/<model-type>/towers_40.pkl
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
