# Blocks Guide

## Simulated Block Set
To generate a simulated block set, run ```python -m learning.domains.towers.create_block_set_file --n-blocks N_BLOCKS --mode random``` where ```N_BLOCKS``` is the number of blocks in the desired set. This file will be saved to ```learning\domains\towers```. Any file that takes in a ```--block-set``` command line argument can use this generated pickle file.

## Physical Block Set

1. **Generate random block set:** Generate the random dimensions of the blocks you would like to construct by running ```python -m scripts.generate_block_set``` in the ```panda_vision``` repo. Change line 69 to ```gen_csv=True```. Change lines 70-77 based on the block materials, desired maximum and minimum block dimensions, and robot gripper constraints. This will result in some output that is necessary for the next step, as well as a csv file that will be used in step 5.
2. **Construct weighted blocks** [TODO]
3. **Generate block tags**: You can generate AR tags or colored tagged to adhere to the blocks. See ```panda_vision/README``` for instructions on how to generate tags.
4. **Adhere tags to blocks**: Once the blocks are constructed and tags are printed and cut down to side, use double-sided sticky tape to adhere the tags onto the blocks. It will be clear which tag corresponds to which dimension, however be sure to follow the right hand rule when attaching the tags. If you make 3D axes with your right hand, your thumb should correspond to Z, index finger to X, and middle finger to Y. For the color tags the following colors and axes correspond: (+X, red), (-X, cyan), (+Y, green), (-Y, yellow), (+Z, blue), (-Z, magenta)
5. **Calibrate blocks:** [ONLY NECESSARY IF USING AR TAGS] See the ```panda_vision/README``` for instructions on how to calibrate the blocks.
6. **Get Mass and Center of Mass**: This is currently a manual process. Use a scale to weigh each block, and manual interaction to estimate where the center of mass is. Enter the values into the csv file from step 1. The COM dimensions are in centimeters, and the mass is in grams.
7. **Generate block set pickle file to be used in ```stacking```**: In ```stacking```, run ```python -m learning.domains.towers.create_block_set_file --csv-file CSV_FILE --mode csv``` where ```CSV_FILE``` is the file output from step 1 and altered by step 6. Any file that takes in a ```--block-set``` command line argument can use this generated pickle file.

### Notes
* All of the block set pickle and csv files that we currently use are in ```learning/domains/towers```. ```learning/domains/towers/block_set_info.txt``` has a brief description of each block set.

## Home Poses Markings
The planning system will assume that blocks start at fixed positions behind the robot. Since these positions are not visible to the cameras, they have been manually specified in ```tamp/misc.py:setup_panda_world```. To aid the operator in resetting the blocks when necessary, we drew outlines of each block's home position on the table and marked the orientations.

We use the following procedure to draw an outline of the table for each block.

1. **Start the robot:** Setup the robot and ensure the ```interface.launch``` file is running.
2. **Move arm to block's home position:** With any empty gripper, run ```python go_to_init_pose.py --show-frames --blocks-file BLOCKS_PKL_FILE --id BLOCK_NUM```, where ```BLOCKS_PKL_FILE``` is the ```.pkl``` file generated above and ```BLOCK_NUM``` is the id of the block whose home position you wish to mark. The system will first create a motion plan and then prompt you to continue. It will then move to the grasp pose with the open gripper. It will pause at this position.
3. **Create block markings:** Put the block on the table in between the gripper with the correct orientation (this will be visualized in PyBullet). With a pencil, outline the 4 corners of the block on the table and mark which sides are facing each way. This will help the operator quickly return blocks to the correct pose.
4. **Reset arm:** Move the arm back to the home position by pressing enter on the keyboard.

Repeat this process for each block in the block set.
