from collections import namedtuple

Node = namedtuple('Node', ['parent', 'children', 'leaf', 'value', 'cost'])
NodeValue = namedtuple('NodeValue', ['tower', 'blocks_remaining'])

class Tree:
    def __init__(self, init_value):
        self.nodes = {0: Node(None, [], False, init_value, -1)}
        self.count = 1
        
    def expand(self, value, cost, parent, leaf):
        self.nodes[self.count] = Node(parent, [], leaf, value, cost)
        self.count += 1
        
    def get_best_node(self, expand=True):
        best_node = None
        best_cost = -1
        for node in self.nodes:
            if self.nodes[node].cost >= best_cost:
                if (expand and not self.nodes[node].leaf) or (not expand):
                    best_node = node
                    best_cost = self.nodes[node].cost
        return best_node, self.nodes[best_node].value
        