import argparse
import pickle
import numpy as np

from learning.active.utils import ActiveExperimentLogger
from learning.domains.towers.generate_tower_training_data import sample_random_tower
from tower_planner import TowerPlanner
from block_utils import Object, get_rotated_block
from tamp.misc import load_blocks


# make datasets for all tower heights with 50/50 constructable not constructable split
# from given block_set
def make_test_dataset(blocks, args, include_index=-1):
    dataset = {}
    num_blocks = [2, 3, 4, 5]
    tp = TowerPlanner()
    
    for k in num_blocks:
        print('Generating '+str(k)+' block towers...')
        key = str(k)+'block'
        towers = []
        tower_block_ids = []
        labels = [] # each tower is labeled with (constructable, stable, pw_stable, cog_stable)
        done = False
        n_constructable = 0
        while True:
            if not blocks:
                tower_blocks = [Object.random() for _ in range(k)]
            else: 
                if include_index < 0:
                    tower_blocks = np.random.choice(blocks, k)
                else:
                    extra_blocks = blocks[:include_index] + blocks[include_index+1:]
                    tower_blocks = [blocks[include_index]] + list(np.random.choice(extra_blocks, k-1, replace=False))
            tower = sample_random_tower(tower_blocks, num_blocks=k)
            rotated_tower = [get_rotated_block(b) for b in tower]
            vec_tower = [block.vectorize() for block in tower]
            
            constructable = tp.tower_is_constructable(rotated_tower)
            stable = tp.tower_is_stable(rotated_tower)
            pw_stable = tp.tower_is_pairwise_stable(rotated_tower)
            cog_stable = tp.tower_is_cog_stable(rotated_tower)
            label = int(constructable)
            block_ids = [block.get_id() for block in rotated_tower]
            if constructable and (n_constructable < args.samples_per_height/2): 
                n_constructable += 1
                labels.append(label)    
                towers.append(vec_tower)
                tower_block_ids.append(block_ids)
            elif (not constructable) and len(towers)-n_constructable < args.samples_per_height/2:
                labels.append(label)
                towers.append(vec_tower)
                tower_block_ids.append(block_ids)
            elif len(towers) == args.samples_per_height:
                break

        
        dataset[key] = {'towers': np.array(towers), 'labels': np.array(labels), 'block_ids': np.array(tower_block_ids)}
        
    with open(args.output_fname, 'wb') as f:
        pickle.dump(dataset, f)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-block-set-fname', 
                        type=str, 
                        default='',
                        help='path to the block set file. if not set, args.n_blocks random blocks generated.')
    parser.add_argument('--eval-block-set-fname', 
                        type=str, 
                        default='',
                        help='path to the block set file. if not set, args.n_blocks random blocks generated.')                    
    parser.add_argument('--samples-per-height',
                        type=int,
                        default=1000)
    parser.add_argument('--output-fname',
                        type=str,
                        required=True)
    parser.add_argument('--debug',
                        action='store_true',
                        help='set to run in debug mode')
    parser.add_argument('--n-blocks',
                        type=int,
                        required=True)
    parser.add_argument('--eval-block-ixs', 
                        nargs='+', 
                        type=int, 
                        default=[], 
                        help='Indices of which eval blocks to use.')
    
    args = parser.parse_args()
    if args.debug:
        import pdb; pdb.set_trace()
 
    if args.train_block_set_fname != '':
        print(args.eval_block_ixs)
        block_set = load_blocks(train_blocks_fname=args.train_block_set_fname, 
                                eval_blocks_fname=args.eval_block_set_fname,
                                eval_block_ixs=args.eval_block_ixs,
                                num_blocks=args.n_blocks)
    else:
        block_set = None
    
    make_test_dataset(block_set, args=args, include_index=10)
    