import argparse
import pickle
import os
import re

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
    
    for tower_file in os.listdir(os.path.join(args.exp_path, 'towers')):
        tower_path_str = r'labeled_tower_(.*)_(.*)_%d_%d.pkl' % (args.tower_number, args.acquisition_step)
        matches = re.match(tower_path_str, tower_file)
        if matches: # sometimes other system files get saved here (eg. .DStore on a mac). don't parse these
            tower_path = matches.group(0)
    
    with open(os.path.join(args.exp_path, 'towers', tower_path), 'rb') as handle:
        tower_data = pickle.load(handle)
        
    print('original label: ', tower_data[2])
    tower_data[2] = args.label
    
    with open(os.path.join(args.exp_path, 'towers', tower_path), 'wb') as handle:
        pickle.dump(tower_data, handle)
    print('new label: ', tower_data[2])
    
