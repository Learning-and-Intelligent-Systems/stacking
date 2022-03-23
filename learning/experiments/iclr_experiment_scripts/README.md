# Steps to Reproduce ICLR Results

## 1. Create the datasets.

1a. Create training datasets: `learning/experiments/iclr_experiment_scripts/make_train_datasets.sh`
1b. Create validation datasets: `learning/experiments/iclr_experiment_scripts/make_val_datasets.sh`
1c. Create testings datasets with fitting blocks:

## 2. Train base models for different block sizes.
`learning/experiments/iclr_experiment_scripts/train_10_block_base_models.sh`
`learning/experiments/iclr_experiment_scripts/train_50_block_base_models.sh`
`learning/experiments/iclr_experiment_scripts/train_100_block_base_models.sh`

## 3. Run PF fitting for each base model on 10 test blocks.
`python -m learning.experiments.pf_fit_all_blocks --pretrained-ensemble-exp-path <PRETRAINED_MODEL> --ensemble-tx 0 --block-set-fname learning/data/iclr_data/blocks/10_random_block_set_2.pkl`     

## 4. Run evaluations.

# Steps to setup a new GCP instance.

1. Clone the repository:
`git clone https://github.com/Learning-and-Intelligent-Systems/stacking.git`
`git checkout latents`

2. Install dependencies:
`virtualenv --python=/usr/bin/python3 .venv`
`source .venv/bin/activate`
`pip install -r requirements.txt`
`pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html`
TODO: Add quick instructions for pddlstream and pbrobot.
`cd ~`
`mkdir external`
`git clone https://github.com/mike-n-7/pb_robot`     
`git clone https://github.com/caelan/pddlstream.git`
`./downward/build.py` 
`cd ~/stacking`
`ln -s ~/external/pb_robot/src/pb_robot .`
`ln -s ~/external/pddlstream/pddlstream .`

3. Copy over dataset and experiments.
`rsync --progress -r learning/data/* michael@35.197.45.205:/home/michael/stacking/learning/data/`
`rsync --progress michael@ruchbah.csail.mit.edu:/home/michael/workspace/stacking_latents/learning/experiments/logs/* stacking/learning/experiments/logs/`
Remove the logs we wont evaluate on heret make copying back cleaner.