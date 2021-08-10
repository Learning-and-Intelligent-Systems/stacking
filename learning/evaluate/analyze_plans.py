import argparse
import numpy as np
import matplotlib.pyplot as plt
import pydot

from learning.active.utils import GoalConditionedExperimentLogger

# learning the delta state saves states as vectors
# all others (including learning a classifier) save the state as a LogicalState
def node_to_label(node):
    if isinstance(node.state, tuple): # vec state
        pstate = str(np.round(node.state[1].squeeze()))
    else: # logical state
        pstate = ''
        if len(node.state.stacked_blocks) > 0:
            pstate += str(node.state.stacked_blocks[0])
            for block_id in node.state.stacked_blocks[1:]:
                pstate += '/'+str(block_id)
    pstate += '\n'+str(node.value)
    return pstate

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
                    #goal = plan_logger.load_plan_goal()
                    #tree = plan_logger.load_plan_tree()
                    #plan = plan_logger.load_final_plan()
                    #dot_graph = generate_dot_graph(tree, plan, goal)
                    print('Planning path %s' % plan_logger.exp_path)
                    model_args = model_logger.load_args()
                    print(model_args.pred_type)
                    #plan_logger.save_dot_graph(dot_graph)
