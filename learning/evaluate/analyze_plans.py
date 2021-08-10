import argparse
import numpy as np
import matplotlib.pyplot as plt
import pydot

from learning.active.utils import GoalConditionedExperimentLogger

# this generally doesn't work, but works when the pred_type == class
# because the edge states are all valid states
def vec_state_to_str(edge_features, action, pred):
    stacked_blocks = set()
    num_objects = edge_features.shape[0]
    for bottom_i in range(num_objects):
        for top_i in range(num_objects):
            if edge_features[bottom_i, top_i] == 1.:
                if bottom_i != 0 and top_i != 0: # 0 is the table
                    stacked_blocks.add(bottom_i)
                    stacked_blocks.add(top_i)

    # string of stack
    str_state = stacked_blocks_to_str(stacked_blocks)
    # add action to string
    #str_state += '  --  %i/%i' % (action[0], action[1])
    # add pred type
    str_state += '  --  %i' % pred
    return str_state

def stacked_blocks_to_str(stacked_blocks):
    stacked_blocks = list(stacked_blocks)
    stacked_blocks.sort()
    str_state = ''
    if len(stacked_blocks) > 0:
        str_state += str(stacked_blocks[0])
        for block_id in stacked_blocks[1:]:
            str_state += '/'+str(block_id)
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
                        print(model_args.pred_type)
                        plan_logger.save_dot_graph(dot_graph)
                    if gen_model_accuracy:
                        dataset_path = model_args.dataset_exp_path
                        dataset_logger = GoalConditionedExperimentLogger(dataset_path)
                        dataset = dataset_logger.load_trans_dataset()
                        dataset.set_pred_type(model_args.pred_type)

                        if dataset.pred_type == 'class': # for now not visualizing delta state transitions
                            # plot training dataset
                            transition_counts = {}
                            transition_actions = {}
                            for (object_features, edge_features, action), pred in dataset:
                                transition = vec_state_to_str(edge_features, action, pred)
                                action_str = ' %i/%i ' % (action[0], action[1])
                                if transition in transition_counts:
                                    transition_counts[transition] += 1
                                    transition_actions[transition].add(action_str)
                                else:
                                    transition_counts[transition] = 1
                                    transition_actions[transition] = set()
                                    transition_actions[transition].add(action_str)

                            fig, ax = plt.subplots()
                            bar_plot = ax.barh(np.arange(len(transition_counts)), transition_counts.values(), align='center')
                            ax.set_yticks(np.arange(len(transition_counts)))
                            ax.set_yticklabels(transition_counts.keys())
                            ax.set_ylabel('Transitions')
                            ax.set_xlabel('Frequency')

                            # show actions on bars
                            for rect, transition in zip(bar_plot, transition_actions):
                                action_text = ','.join(list(transition_actions[transition]))
                                print(action_text)
                                ax.text(0,
                                        rect.get_y() + rect.get_height()/2,
                                        action_text,
                                        ha='left',
                                        va='center',
                                        rotation=0)

                            plt.show()


                        # plot accuracy in different test domains
