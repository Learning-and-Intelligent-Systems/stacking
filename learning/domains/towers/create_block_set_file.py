import csv
import pickle
import numpy as np
import argparse

from block_utils import Object, Dimensions, Color, Position

# this is used to generate a block_set.pkl file from random block_utils.Object() 
# blocks using the parameters there
def random_block_set(args):
    block_set = [Object.random('obj_'+str(n)) for n in range(args.n_blocks)]
    pkl_filename = 'block_set_'+str(args.n_blocks)+'.pkl'
    with open(pkl_filename, 'wb') as f:
        pickle.dump(block_set, f)

def string_to_list(string):
    string = string.replace('[', '').replace(']','')
    list_string = string.split(' ')
    list_float = [float(s) for s in list_string if s != '']
    return list_float

# this is used to generate a block_set.pkl file from a .csv file 
# the .csv files are generated in scripts/generate_block_set.py (for physical block sets)
def block_set_from_csv(args):
    block_set = []
    with open(args.csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row_i, row in enumerate(csv_reader):
            if row_i > 0:
                id, dimensions, com, mass = row
                
                block = Object('obj_'+id,
                                Dimensions(*np.multiply(string_to_list(dimensions), .01)), # g --> kg
                                float(mass)*.001, # g --> kg
                                Position(*np.multiply(string_to_list(com), .01)), # cm --> m
                                Color(*np.random.rand(3)))
                print(block.dimensions, block.mass, block.com)
                block_set.append(block)
            
    pkl_filename = args.csv_file[:-4]+'.pkl'
    with open(pkl_filename, 'wb') as f:
        pickle.dump(block_set, f)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-file', type=str)
    parser.add_argument('--n-blocks', type=int)
    parser.add_argument('--mode', choices=['csv', 'random'])
    args = parser.parse_args()

    if args.mode == 'random':
        random_block_set(args)
    elif args.mode == 'csv':
        block_set_from_csv(args)