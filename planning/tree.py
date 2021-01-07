from collections import namedtuple
import numpy as np

Node = namedtuple('Node', ['parent', 'children', 'term', 'leaf', 'value', 'count', \
                'tower', 'blocks_remaining', 'tower_height', 'ground_truth'])

class Tree:
    def __init__(self, blocks):
        self.nodes = {0: {'parent': None, 
                            'children': [],
                            'term': False,
                            'leaf': True,
                            'value': 0, 
                            'count': 0,
                            'tower': [],
                            'blocks_remaining': blocks,
                            'tower_height': 0,
                            'ground_truth': 0}}
        self.count = 1
        
    def expand(self, node):
        self.nodes[self.count] = node
        self.nodes[node['parent']]['children'] += [self.count]
        self.nodes[node['parent']]['leaf'] = False
        self.count += 1
        
    def get_exp_best_node_expand(self):
        node_rewards = np.array([node['value'] for node in self.nodes.values()])
        # if no node has a reward then they all have equal prob of being selected
        if sum(node_rewards) == 0:
            node_rewards = np.ones(len(node_rewards))
        norm_node_rewards = node_rewards/np.sum(node_rewards)
        valid_next_node = False
        while not valid_next_node:
            next_node_i = np.random.choice(list(self.nodes.keys()), p=norm_node_rewards)
            next_node_id = list(self.nodes.keys())[next_node_i]
            if not self.nodes[next_node_id]['term']:
                valid_next_node = True
        return next_node_id
    
    def get_exp_best_node(self, height):
        best_node = None
        best_exp_reward = -1
        for node in self.nodes:
            if (self.nodes[node].exp_reward >= best_exp_reward) and \
                    (len(self.nodes[node].value.tower) == height):
                best_node = node
                best_exp_reward = self.nodes[node].exp_reward
        return best_node
        
    def get_ground_truth_best_node(self, height):
        best_node = None
        best_gt_reward = -1
        for node in self.nodes:
            if (self.nodes[node].ground_truth >= best_gt_reward) and \
                    (len(self.nodes[node].value.tower) == height):
                best_node = node
                best_gt_reward = self.nodes[node].ground_truth
        return best_node