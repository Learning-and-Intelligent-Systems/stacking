from collections import namedtuple
import numpy as np

Node = namedtuple('Node', ['parent', 'children', 'term', 'value', 'exp_reward', 'reward', 'ground_truth'])
NodeValue = namedtuple('NodeValue', ['tower', 'blocks_remaining'])

class Tree:
    def __init__(self, init_value):
        self.nodes = {0: Node(None, [], False, init_value, 0, 0, 0)}
        self.count = 1
        
    def expand(self, value, exp_reward, reward, ground_truth, parent, term):
        self.nodes[self.count] = Node(parent, [], term, value, exp_reward, reward, ground_truth)
        self.count += 1
        
    def get_next_node(self):
        node_rewards = np.array([node.exp_reward for _, node in self.nodes.items()])
        # if no node has a reward then they all have equal prob of being selected
        if sum(node_rewards) == 0:
            node_rewards = np.ones(len(node_rewards))
        norm_node_rewards = node_rewards/np.sum(node_rewards)
        valid_next_node = False
        while not valid_next_node:
            next_node_i = np.random.choice(list(self.nodes.keys()), p=norm_node_rewards)
            next_node_id = list(self.nodes.keys())[next_node_i]
            if not self.nodes[next_node_id].term:
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