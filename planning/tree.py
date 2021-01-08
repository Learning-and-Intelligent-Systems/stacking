from collections import namedtuple
import numpy as np


class Tree:
    def __init__(self, blocks):
        self.nodes = {0: {'parent': None, 
                            'children': [],
                            'term': False,
                            'leaf': True,
                            'exp_reward': 0,
                            'value': 0, 
                            'count': 0,
                            'tower': [],
                            'blocks_remaining': blocks,
                            'tower_height': 0,
                            'ground_truth': 0}}
        self.count = 1
        
    def traverse(self, c):
        node_id = 0
        while not self.nodes[node_id]['leaf']:
            node_id = self.get_uct_node(node_id, c)
        if self.nodes[node_id]['term']:
            return self.nodes[node_id]['parent']
        return node_id
    
    # random rollout
    def rollout(self, node_id, problem, model):
        node = self.nodes[node_id]
        if not node['term']:
            # TODO: should make a seprate function that just returns one action
            new_nodes = problem.sample_actions(self.nodes[node_id], model)
            node = np.random.choice(new_nodes)
        return node['exp_reward']
    
    # adjust value and increase count
    def backpropagate(self, node_id, value):
        while not node_id == 0: # while not at root node
            old_count = self.nodes[node_id]['count']
            old_value = self.nodes[node_id]['value']
            new_value = (old_count*old_value + value)/(old_count+1)
            self.nodes[node_id]['count'] += 1
            self.nodes[node_id]['value'] = new_value
            
            node_id = self.nodes[node_id]['parent']
        
    def get_uct_node(self, node_id, c):
        uct_values = np.zeros(len(self.nodes[node_id]['children']))
        for i, child_id in enumerate(self.nodes[node_id]['children']):
            node_value = self.nodes[child_id]['value']
            expl_bonus =  c*np.sqrt(np.log(self.nodes[node_id]['count'])/self.nodes[child_id]['count'])
            uct_values[i] = node_value + expl_bonus
        uct_node_index = np.argmax(uct_values)
        return self.nodes[node_id]['children'][uct_node_index]
        
    def expand(self, parent_node_id, child_node):
        child_node_id = self.count
        child_node['parent'] = parent_node_id
        self.nodes[child_node_id] = child_node
        self.nodes[parent_node_id]['children'] += [child_node_id]
        self.nodes[parent_node_id]['leaf'] = False
        self.count += 1
        return child_node_id
        
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