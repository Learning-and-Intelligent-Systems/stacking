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
* `datasets` : The full dataset used to train all models 
* `val_datasets` : validation datasets used on all models
* `figures` : Contains validation and/or analysis plots generated for this experiment by other auxiliary scripts
* `args.pkl` : Set of arguments used for restarting training
* `block_placement_data.pkl` : Information on how many blocks were placed before the tower fell for each tower

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

If a tower is mislabeled during acquisition step ```<tx>```, run the following AFTER all towers for step ```<tx>``` have been labeled, and ```acquired_<tx>.pkl``` has been generated.
```<tn>``` corresponds to the tower number integer in the towers file name (0 indexed).
 ```<l>``` is the correct label.
If any files from step ```<tx>+1``` have  been generated then delete them. (They will be regenerated/trained on restart)
```
python3 -m learning.experiments.fix_tower_label --exp-path <path> --acquisition-step <tx>
--tower-number <tn> --label <l>
```
---

## Monitor Accuracy of Models Trained with the Panda

To evaluate current training progress when training on a Panda robot, run this with the appropriate arguments.

```
python3 -m learning.evaluate.plot_model_accuracy --exp-paths learning/experiments/logs/<experiment-name> --max-acquisitions <tx> --plot-step 1 --test-set-fname learning/evaluate/test_datasets/eval_blocks_test_dataset.pkl --output-fname <output-img>
```

```<tx>``` is the index of the last executed acquisition step.

This will output an accuracy plot to ```<output-img>.png```.

---

## Evaluate Task Performance of a Trained Model

Finally, you can execute a task using the model trained on the simulated or real Panda robot. In simulation, we calculate regret, and output this data to pickle files in ```learning/experiments/logs/<experiment-name>/figures```. The next section will discuss how to convert these files into plots. On the real panda, regret is calculated manually (?TODO?) and qualitative and quantitative results are generated manually.

### Evaluate Task Performance with the Panda

To generate a set of evaluation towers from the learned model for the Panda to construct:
```
python3 -m learning.evaluate.plan_evaluate_models --problem <problem> --block-set-fname learning/domains/towers/eval_block_set_9.pkl --exp-path learning/experiments/logs/<experiment-name> --acquisition-step <acquisition-step> --n-towers <n-towers>
```
Where ```<problem>``` can be one of: ['tallest', 'overhang', 'min-contact', 'cumulative-overhang'], ```<acquisition-step>``` is the acquisition step of the model you would like to use (most likely the last one trained), ```<n-towers>``` is how many evaluation towers you would like the robot to build, ```<planning-mode>``` is one of ['learned', 'simple-model', 'noisy-model']. If using the 'noisy-model' you must also add ```--plan-xy-noise <planning-model-noise>``` where ```<plan-xy-noise>``` is a float representing the variance of the Gaussian noise added to block placements.

This will output a file to ```learning/experiments/logs/<experiment-name>/evaluation_towers/<problem>/<planning-model>/towers_<n-towers>.pkl``` 

If you run the same command more than once, towers will be appended to the existing tower files.

To build the towers on the panda robot:
```
python3 -m learning.experiments.run_towers_evaluation --real --use-vision --use-planning-server --blocks-file learning/domains/towers/eval_block_set_9.pkl --towers-file learning/experiments/logs/<experiment-name>/evaluation_towers/cumulative-overhang/learned/towers_<n-towers>.pkl
```

### Evaluate Task Performance with the Simulator

With the robot evaluations we only evaluate a single acquisition step at a time, but in simulation we can evaluate over many acquisition steps with a single command. This step will save performance data to a .pkl file which will be used later to generate figures. Use the following:
```
python3 -m learning.evaluate.plan_evaluate_models --problem <problem> --block-set-fname learning/domains/towers/eval_sim_block_set_10.pkl --exp-path learning/experiments/logs/<experiment-name> --max-acquisitions <max-acquisitions> --n-towers <n-towers> --exec-mode <exec-mode> --planning-model <planning-model> --exec-xy-noise <exec-xy-noise> --plan-xy-noise <plan-xy-noise>
```
```<problem>``` can be one of: ['tallest', 'overhang', 'min-contact', 'cumulative-overhang'].

```<max-acquisitions>``` is the final acquisition step to be evaluated. Each model from 0 to this number in increments of 10 will be evaluated.

```<n-towers>``` is how evaluation many towers will be evaluated for each acquisition step.

```<exec-mode>``` can be one of ['simple-model', 'noisy-model']. When building the towers found by the planner, 'simple-model' adds no noise, while 'noisy-model' adds ```<exec-xy-noise>``` Gaussian noise.

```<planning-model>``` can be one of ['learned', 'noisy-model', 'simple-model']. When planning, the generated towers can evaluate stability using the 'learned' model, or a 'simple-model' which uses an analytical model of stability, or a 'noisy-model' which uses the analytical model, but adds ```<plan-xy-noise>``` Gaussian noise when assessing stability.

---
## Plot Simulation Results

### Model Accuracy

Make an evaluation dataset:

```
python3 -m learning.evaluate.make_test_datasets  --block-set-fname learning/domains/towers/eval_sim_block_set_10.pkl --output-fname <dataset-output>
```
```<dataset-output>``` is the name of the file to save the dataset to.

To plot the accuracy of a model trained in simulation:
```
python3 -m learning.evaluate.plot_model_accuracy --exp-paths learning/experiments/logs/<experiment-name> --max-acquisitions <tx> --plot-step <ps> --test-set-fname <dataset-output> --output-fname <output-img>
```

```<tx>``` is the index of the last executed acquisition step.

```<ps>``` is the number of acquisitions steps between each plotted point (set ```ps=1``` for the highest resolution).

This will output an accuracy plot to ```<output-img>.png```.


To plot the accuracy of several different methods, each used to train several models in simulation, you have to go into ```learning/evaluate/plot_compare_model_accuracy.py``` and change the ```exp_paths``` variable. It is a dictionary where they key corresponds to a method (and the word that will appear in the plot legend), and the key is a list of paths to models of the same type that will be aggregated. Then run the following:
```
```

This will output an accuracy plot to ```<output-img>.png```.

### Task Performance

To plot task performance, you have to have generated the pickle files storing the regret results (see above section "Evaluate Task Performance with the Simulator"). Then to generate plots for comparing task performance, go into ```learning/evaluate/plot_combine_plan_evaluate_models``` and change the ```y_axis``` variable to the metric you want to plot (they are listed there). Also change the ```exp_paths``` dictionary to the paths to experiments you want to compare. The keys are the methods (will show up in the legend of the final plot), and the values are lists of file paths to the experiments of the same type. The results will be aggregated across all experiments in the list. The ```color_map``` indicates what color (as a hex color code) you want each method to show up as in the final plot. Make sure it is the same length as the ```exp_paths``` dictionary.

```
python3 -m learning.evaluate.plot_combine_plan_evaluate_models --max-acquisitions <tx> --problems <problems>
```

This will plot performance for every 10 acquisition steps from 0 to ```<tx>``` and save the resulting plot in ```stacking/figures/```.

```<problems>``` is a list of problems to generate plot for: ['tallest', 'min_contact', 'max_overhang']

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
