import argparse
import pickle
import os
import numpy as np

from learning.active.utils import ActiveExperimentLogger

import pdb; pdb.set_trace()

tx = 13
exp_path = 'learning/experiments/logs/robot-seq-init-sim-20210219-131924'
logger = ActiveExperimentLogger(exp_path)
towers = logger.get_towers_data(tx)
new_data = {'%dblock' % nblocks: {} for nblocks in [2, 3, 4, 5]}
for k in new_data:
    new_data[k] = {'towers': [], 'labels': [], 'block_ids': []}
for tower_data in towers:
    tower, block_ids, labels = tower_data
    size = len(tower)
    new_data['%dblock' % size]['towers'].append(tower)
    new_data['%dblock' % size]['block_ids'].append(block_ids)
    new_data['%dblock' % size]['labels'].append(labels)

for k in new_data:
    new_data[k]['towers'] = np.array(new_data[k]['towers'])
    new_data[k]['block_ids'] = np.array(new_data[k]['block_ids'])
    new_data[k]['labels'] = np.array(new_data[k]['labels'])

logger.save_acquisition_data(new_data, None, tx)
