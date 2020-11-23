# configure minio
./mc config host add honda_cmm 	https://ceph.csail.mit.edu 526add252055448fbf4b7bb711b822c7 84eaef04916848c1b061a5362f3d98ac

# clone stacking repo
git clone --single-branch --branch visual_learning https://github.com/Learning-and-Intelligent-Systems/stacking.git

# copy datasets from minio to container
/./mc cp honda_cmm/stacking/random_blocks_\(x1600\)_2to5blocks_uniform_density_relative_pos.pkl stacking/learning/data
/./mc cp honda_cmm/stacking/random_blocks_\(x10000\)_2to5blocks_uniform_density_relative_pos.pkl stacking/learning/data

# link other packages
cd stacking
ln -s /pb_robot/src/pb_robot .
ln -s /pddlstream/pddlstream .

# run training code
python3.7 -m learning.train_iter

# copy over results to minio
export TIMESTAMP=`date '+%F_%H:%M:%S'`
#/./mc cp train_losses.png honda_cmm/stacking/${TIMESTAMP}/train_losses.png
#/./mc cp test_accuracies.png honda_cmm/stacking/${TIMESTAMP}/test_accuracies.png
/./mc cp test_accuracies_2blocks.png honda_cmm/stacking/${TIMESTAMP}/test_accuracies_2blocks.png
/./mc cp test_accuracies_3blocks.png honda_cmm/stacking/${TIMESTAMP}/test_accuracies_3blocks.png
/./mc cp test_accuracies_4blocks.png honda_cmm/stacking/${TIMESTAMP}/test_accuracies_4blocks.png
/./mc cp test_accuracies_5blocks.png honda_cmm/stacking/${TIMESTAMP}/test_accuracies_5blocks.png
/./mc cp results.pickle honda_cmm/stacking/${TIMESTAMP}/params.txt
/./mc cp params.txt honda_cmm/stacking/${TIMESTAMP}/results.pickle
/./mc cp model0.pt honda_cmm/stacking/${TIMESTAMP}/model0.pt
/./mc cp model1.pt honda_cmm/stacking/${TIMESTAMP}/model1.pt
/./mc cp model2.pt honda_cmm/stacking/${TIMESTAMP}/model2.pt
