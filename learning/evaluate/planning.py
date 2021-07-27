import argparse
import numpy as np
import matplotlib.pyplot as plt

from tamp.predicates import On
from planning import plan
from learning.domains.abc_blocks.world import ABCBlocksWorldGT, print_state
from learning.active.utils import GoalConditionedExperimentLogger

def generate_random_goal(world):
    top_block_num = np.random.randint(world.min_block_num+1, world.max_block_num+1)
    return [On(top_block_num-1, top_block_num)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name',
                        type=str,
                        required=True,
                        help='where to save exp data')
    parser.add_argument('--search-mode',
                        type=str,
                        default='mcts',
                        choices=['mcts', 'heuristic'],
                        help='What type of search, mcts rollouts or a best-first with a learned heuristic')
    parser.add_argument('--debug',
                        action='store_true',
                        help='set to run in debug mode')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()
    #try:
    num_goals = 10
    num_plan_attempts = 1
    all_plan_num_blocks = [2, 3, 4, 5]#, 6, 7, 8]

    model_paths = ['learning/experiments/logs/models/2_random-20210727-093617',
                    'learning/experiments/logs/models/3_random-20210727-100817',
                	'learning/experiments/logs/models/4_random-20210727-100851',
                    'learning/experiments/logs/models/5_random-20210727-100337']

    plan_args = argparse.Namespace(num_branches=10,
                        timeout=100,
                        c=.01,
                        max_ro=10,
                        model_type='learned',
                        search_mode=args.search_mode)

    success_data = {}
    plan_paths = {}

    for model_path in model_paths:
        plan_args.model_exp_path = model_path
        model_logger = GoalConditionedExperimentLogger(model_path)
        model_args = model_logger.load_args()
        model_num_blocks = model_args.num_blocks
        success_data[model_num_blocks] = {}
        plan_paths[model_num_blocks] = {}
        for plan_num_blocks in all_plan_num_blocks:
            success_data[model_num_blocks][plan_num_blocks] = []
            plan_paths[model_num_blocks][plan_num_blocks] = []
            plan_args.num_blocks = plan_num_blocks
            world = plan.setup_world(plan_args)
            for _ in range(num_goals):
                goal = generate_random_goal(world)
                for _ in range(num_plan_attempts):
                    plan_args.exp_name = 'train_blocks_%i_policy_%s_plan_blocks_%i' % (model_args.num_blocks, model_args.policy, plan_num_blocks)
                    found_plan, plan_exp_path = plan.run(goal, plan_args)
                    if found_plan:
                        test_world = ABCBlocksWorldGT(plan_num_blocks)
                        final_state = test_world.execute_plan([node.action for node in found_plan])
                        success = test_world.is_goal_state(final_state, goal)
                    else:
                        success = False
                    print_state(goal, world.num_objects)
                    success_data[model_num_blocks][plan_num_blocks].append(success)
                    plan_paths[model_num_blocks][plan_num_blocks].append(plan_exp_path)

    cs = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    figure, axis = plt.subplots()
    for i, (model_num_blocks, success_model_data) in enumerate(success_data.items()):
        success_plot_data = [np.sum(success_model_data[plan_num_blocks])/(num_goals*num_plan_attempts) for plan_num_blocks in all_plan_num_blocks]
        axis.plot(all_plan_num_blocks, success_plot_data, color=cs[i], label=str(model_num_blocks))

    axis.set_xticks(all_plan_num_blocks)
    axis.set_title('Planning Performance with Learned Models')
    axis.set_xlabel('Number of Planning Blocks')
    axis.set_ylabel('% Success')
    axis.legend(title='Number of Training Blocks')

    logger = GoalConditionedExperimentLogger.setup_experiment_directory(args, 'eval_goals')
    plt.savefig(logger.exp_path+'/eval_goals.png')
    logger.save_plot_data([plan_paths, success_data])
    print('Saving figures and data to %s.' % logger.exp_path)
    #except:
    #    import pdb; pdb.post_mortem()
