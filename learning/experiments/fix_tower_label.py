import argparse
import pickle
import os
import re
import numpy as np

from learning.active.utils import ActiveExperimentLogger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-path',
                        type=str,
                        required=True)
    parser.add_argument('--acquisition-step', 
                        type=int, 
                        required=True)
    parser.add_argument('--tower-number',
                        type=int,
                        required=True)
    parser.add_argument('--label',
                        type=float,
                        required=True)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()
        
    logger = ActiveExperimentLogger(args.exp_path)
    
    # regenerate erroneous tower file
    for tower_file in os.listdir(os.path.join(args.exp_path, 'towers')):
        tower_path_str = r'labeled_tower_(.*)_(.*)_%d_%d.pkl' % (args.tower_number, args.acquisition_step)
        matches = re.match(tower_path_str, tower_file)
        if matches: # sometimes other system files get saved here (eg. .DStore on a mac). don't parse these
            tower_path = matches.group(0)
    
    input('Press enter to edit file: %s' % tower_path)
    with open(os.path.join(args.exp_path, 'towers', tower_path), 'rb') as handle:
        tower_data = pickle.load(handle)
        
    print('original label: ', tower_data[2])
    tower_data[2] = args.label
    
    with open(os.path.join(args.exp_path, 'towers', tower_path), 'wb') as handle:
        pickle.dump(tower_data, handle)
    print('new label: ', tower_data[2])
    
    # regenerate acquired data from tower files
    acquired_towers = logger.get_towers_data(args.acquisition_step)
    new_data = {'%dblock' % nblocks: {} for nblocks in [2, 3, 4, 5]}
    for k in new_data:
        new_data[k] = {'towers': [], 'labels': [], 'block_ids': []}
    for tower_data in acquired_towers:
        tower, block_ids, labels = tower_data
        size = len(tower)
        new_data['%dblock' % size]['towers'].append(tower)
        new_data['%dblock' % size]['block_ids'].append(block_ids)
        new_data['%dblock' % size]['labels'].append(labels)

    for k in new_data:
        new_data[k]['towers'] = np.array(new_data[k]['towers'])
        new_data[k]['block_ids'] = np.array(new_data[k]['block_ids'])
        new_data[k]['labels'] = np.array(new_data[k]['labels'])

    print('Regenerating acquired_%d.pkl with fixed tower label' % args.acquisition_step)
    
    # make acquired_processing so logger doesn't complain when it tries to remove it
    with open(os.path.join(args.exp_path, 'acquired_processing.pkl'), 'wb') as handle:
        pickle.dump({}, handle)
    logger.save_acquisition_data(new_data, None, args.acquisition_step)

