import argparse
import numpy as np
import matplotlib.pyplot as plt
import pydot

from learning.domains.abc_blocks.world import ABCBlocksWorldGT, LogicalState
from learning.active.utils import GoalConditionedExperimentLogger

# this generally doesn't work, but works when the pred_type == class
# because the edge states are all valid states
def vec_to_logical_state(edge_features, world):
    state = LogicalState(world._blocks, world.num_blocks, world.table)
    state.stacked_blocks = vec_state_to_stacked_blocks(edge_features)
    return state

def vec_state_to_stacked_blocks(edge_features):
    stacked_blocks = set()
    num_objects = edge_features.shape[0]
    for bottom_i in range(num_objects):
        for top_i in range(num_objects):
            if edge_features[bottom_i, top_i] == 1.:
                if bottom_i != 0 and top_i != 0: # 0 is the table
                    stacked_blocks.add(bottom_i)
                    stacked_blocks.add(top_i)
    stacked_blocks = list(stacked_blocks)
    stacked_blocks.sort()
    return stacked_blocks

def stacked_blocks_to_str(stacked_blocks):
    if len(stacked_blocks) > 0:
        str_state = ''
        str_state += str(stacked_blocks[0])
        for block_id in stacked_blocks[1:]:
            str_state += '/'+str(block_id)
    else:
        str_state = '-'
    return str_state

# learning the delta state saves states as vectors
# all others (including learning a classifier) save the state as a LogicalState
def node_to_label(node):
    if isinstance(node.state, tuple): # vec state
        str_state = str(np.round(node.state[1].squeeze())) + 0 # get rid of -0s
    else: # logical state
        stacked_blocks_to_str(node.state.stacked_blocks)
    str_state += '\n'+str(node.value)
    return str_state

def generate_dot_graph(tree, plan, goal):
    plan_ids = []
    if plan is not None:
        plan_ids = [node.id for node in plan]
    graph = pydot.Dot("my_graph", graph_type="graph")
    # add goal node TODO: make more general
    graph.add_node(pydot.Node(-1, label=str(goal[0].bottom_num)+'/'+str(goal[0].top_num), color='red'))
    # make all nodes
    for node_id, node in tree.nodes.items():
        label = node_to_label(node)
        if node_id in plan_ids:
            graph.add_node(pydot.Node(node_id, label=label, color="green"))
        else:
            graph.add_node(pydot.Node(node_id, label=label))
    # add all edges
    for node_id, node in tree.nodes.items():
        for child_id in node.children:
            graph.add_edge(pydot.Edge(node_id, child_id))
    return graph

def plot_horiz_bars(transitions, plot_inds, bar_text_inds):
    from operator import itemgetter
    def join_strs(str_list, indices):
        if len(indices) == 1:
            return str_list[indices[0]]
        else:
            return ' , '.join(itemgetter(*indices)(str_list))

    plot_data = {}
    plot_text = {}
    for transition in transitions:
        key = join_strs(transition, plot_inds)
        bar_text = join_strs(transition, bar_text_inds)
        if key in plot_data:
            plot_data[key] += 1
            plot_text[key].add(bar_text)
        else:
            plot_data[key] = 1
            plot_text[key] = set()
            plot_text[key].add(bar_text)

    fig, ax = plt.subplots()
    bar_plot = ax.barh(np.arange(len(plot_data)), plot_data.values(), align='center')
    ax.set_yticks(np.arange(len(plot_data)))
    ax.set_yticklabels(plot_data.keys())
    ax.set_ylabel('Transitions')
    ax.set_xlabel('Frequency')

    # show text on bars
    for rect, text_set in zip(bar_plot, plot_text.values()):
        ax.text(0,
                rect.get_y() + rect.get_height()/2,
                ' // '.join(list(text_set)),
                ha='left',
                va='center',
                rotation=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-path',
                        type=str,
                        required=True,
                        help='what planning run to be analyzed')
    parser.add_argument('--debug',
                        action='store_true',
                        help='set to run in debug mode')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    gen_dot_graphs = False
    gen_model_accuracy = True

    logger = GoalConditionedExperimentLogger(args.exp_path)

    plan_plot_data = logger.load_plot_data()

    plot_data_paths = plan_plot_data[0]
    plot_data_successes = plan_plot_data[1]

    for method, method_data in plot_data_paths.items():
        for plan_num_blocks, num_blocks_data in method_data.items():
            for model_path, plan_paths in num_blocks_data.items():
                for plan_path in plan_paths:
                    plan_logger = GoalConditionedExperimentLogger(plan_path)
                    model_logger = GoalConditionedExperimentLogger(model_path)
                    print('Planning path %s' % plan_logger.exp_path)
                    model_args = model_logger.load_args()
                    if gen_dot_graphs:
                        goal = plan_logger.load_plan_goal()
                        tree = plan_logger.load_plan_tree()
                        plan = plan_logger.load_final_plan()
                        dot_graph = generate_dot_graph(tree, plan, goal)
                        #print(model_args.pred_type)
                        plan_logger.save_dot_graph(dot_graph)
                    if gen_model_accuracy:
                        dataset_path = model_args.dataset_exp_path
                        dataset_logger = GoalConditionedExperimentLogger(dataset_path)
                        dataset = dataset_logger.load_trans_dataset()
                        dataset.set_pred_type(model_args.pred_type)

                        dataset_args = dataset_logger.load_args()
                        world = ABCBlocksWorldGT(dataset_args.num_blocks)

                        if dataset.pred_type == 'class': # for now not visualizing delta state transitions
                            # plot training dataset
                            transitions = []
                            for (object_features, edge_features, action), pred in dataset:
                                action = [int(a) for a in action]
                                state = vec_to_logical_state(edge_features, world)
                                next_state = world.transition(state, action)
                                next_opt_state = world.transition(state, action, optimistic=True)

                                str_state = stacked_blocks_to_str(state.stacked_blocks)
                                str_action = '%i/%i' % (action[0], action[1])
                                str_next_state = stacked_blocks_to_str(next_state.stacked_blocks)
                                str_pred = '%i' % pred
                                str_opt_next_state = stacked_blocks_to_str(next_opt_state.stacked_blocks)
                                transitions.append([str_state, str_action, str_next_state, str_pred, str_opt_next_state])

                            all_trans_keys = [0,1,2,3]
                            all_trans_bar_text = [4]
                            plot_horiz_bars(transitions, all_trans_keys, all_trans_bar_text)

                            classification_keys = [3]
                            classification_bar_text = [0,1,2]
                            plot_horiz_bars(transitions, classification_keys, classification_bar_text)

                            init_state_keys = [0,3]
                            init_state_bar_text = [1,2]
                            plot_horiz_bars(transitions, init_state_keys, init_state_bar_text)

                        plt.show()
                        # plot accuracy in different test domains
