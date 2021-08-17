import argparse
import numpy as np
import pydot

from learning.active.utils import GoalConditionedExperimentLogger
from learning.evaluate.utils import stacked_blocks_to_str
# learning the delta state saves states as vectors
# all others (including learning a classifier) save the state as a LogicalState
def state_to_label(state, value):
    if isinstance(state, tuple): # vec state
        str_state = str(state[1].squeeze()) # +0 to get rid of -0s
    else: # logical state
        str_state = stacked_blocks_to_str(state.stacked_blocks)
    str_state += '\n'+str(value)
    return str_state

def generate_dot_graph(plan_path):
    plan_logger = GoalConditionedExperimentLogger(plan_path)
    plan_args = plan_logger.load_args()
    if plan_args.model_type == 'learned':
        model_path = plan_args.model_exp_path
        model_logger = GoalConditionedExperimentLogger(model_path)
        model_args = model_logger.load_args()
    goal = plan_logger.load_plan_goal() # list of predicates
    tree = plan_logger.load_plan_tree()
    plan = plan_logger.load_final_plan()

    plan_ids = []
    if plan is not None:
        plan_ids = [node.id for node in plan]

    title = ''
    if plan_args.model_type == 'learned':
        title = model_args.pred_type+'\n'+model_path+'\n'+model_args.dataset_exp_path
    graph = pydot.Dot("my_graph", graph_type="graph", label=title, labelloc="t")
    ## add goal node TODO: make more general
    from learning.domains.abc_blocks.world import logical_to_vec_state
    from learning.evaluate.utils import vec_to_logical_state
    goal_state = logical_to_vec_state(goal, tree.world.num_objects)
    lgoal_state = vec_to_logical_state(goal_state[1], tree.world)
    graph.add_node(pydot.Node(-1, label=state_to_label(lgoal_state, 1.0), color='red'))
    ##

    # make all nodes
    for node_id, node in tree.nodes.items():
        label = state_to_label(node.state, node.value)
        if node_id in plan_ids:
            graph.add_node(pydot.Node(node_id, label=label, color="green"))
        else:
            graph.add_node(pydot.Node(node_id, label=label))
    # add all edges
    for node_id, node in tree.nodes.items():
        for child_id in node.children:
            graph.add_edge(pydot.Edge(node_id, child_id))
    plan_logger.save_dot_graph(graph)
    print('Saved dot graph to %s' % plan_logger.exp_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-path',
                        type=str,
                        help='what planning run to be analyzed in /exp_goals, else use given hard coded list below')
    parser.add_argument('--debug',
                        action='store_true',
                        help='set to run in debug mode')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    plan_paths = ['learning/experiments/logs/planning/test-20210816-163713']

    if args.exp_path:
        logger = GoalConditionedExperimentLogger(args.exp_path)
        plan_plot_data = logger.load_plot_data()
        plot_data_paths = plan_plot_data[0]
        plot_data_successes = plan_plot_data[1]
        for method, method_data in plot_data_paths.items():
            for plan_num_blocks, num_blocks_data in method_data.items():
                for model_path, plan_paths in num_blocks_data.items():
                    for plan_path in plan_paths:
                        print('Generating dot graphs for plans in %s' % args.exp_path)
                        generate_dot_graph(plan_path)
    else:
        for plan_path in plan_paths:
            print('Generating dot graphs for hardcoded plan_paths in file.')
            generate_dot_graph(plan_path)
