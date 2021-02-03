import csv
import pickle
import numpy as np

from block_utils import Object, Dimensions, Color, Position

def string_to_list(string):
    string = string.replace('[', '').replace(']','')
    list_string = string.split(' ')
    list_float = [float(s) for s in list_string if s != '']
    return list_float

def main():
    block_set = []
    with open('learning/domains/towers/final_block_params.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row_i, row in enumerate(csv_reader):
            if row_i > 0:
                id, dimensions, com, mass = row
                
                block = Object('obj_'+id,
                                Dimensions(*string_to_list(dimensions)),
                                float(mass),
                                Position(*string_to_list(com)),
                                Color(*np.random.rand(3)))
                block_set.append(block)
            
    with open('learning/domains/towers/final_block_set_12.pkl', 'wb') as f:
        pickle.dump(block_set, f)
        
if __name__ == '__main__':
    main()