from block_utils import *
from stability import find_tallest_tower

def create_blocks():
	pass

def experiment(blocks):
	pass

def display_tower(tallest_tower):
	pass

if __name__ == '__main__':
	blocks = create_blocks()

	com_filters = experiment(blocks)

	_, tallest_tower = find_tallest_tower(blocks, com_filters)

	diplay_tower(tallest_tower)
