import csv
import pickle
import numpy as np
import argparse

from block_utils import Object, Dimensions, Color, Position

def string_to_list(string):
    string = string.replace('[', '').replace(']','')
    list_string = string.split(' ')
    list_float = [float(s) for s in list_string if s != '']
    return list_float

def main(args):
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
    parser.add_argument('--csv-file', type=str, required=True)
    args = parser.parse_args()

    main(args)