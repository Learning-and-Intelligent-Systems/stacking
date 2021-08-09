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
    num_goals = 3
    num_plan_attempts = 1
    all_plan_num_blocks = [2]#, 3, 4, 5, 6]#, 7, 8]

    model_paths = {'FULL': ['learning/experiments/logs/models/4_random_delta_state-20210728-112822',
                        'learning/experiments/logs/models/4_random_delta_state-20210728-112837',
                        'learning/experiments/logs/models/4_random_delta_state-20210728-112848',
                        'learning/experiments/logs/models/4_random_delta_state-20210728-112859'],
                	'CLASS': ['learning/experiments/logs/models/4_random_class-20210728-113022',
                        'learning/experiments/logs/models/4_random_class-20210728-113100',
                        'learning/experiments/logs/models/4_random_class-20210728-113110',
                        'learning/experiments/logs/models/4_random_class-20210728-113121']}

    plan_args = argparse.Namespace(num_branches=10,
                        timeout=100,
                        c=.01,
                        max_ro=10,
                        model_type='learned',
                        search_mode=args.search_mode)

    success_data = {}
    plan_paths = {}

    # generate random goals to evaluate planner with
    test_goals = {}
    for plan_num_blocks in all_plan_num_blocks:
        plan_args.num_blocks = plan_num_blocks
        plan_args.model_exp_path = model_paths['FULL'][0]
        world = plan.setup_world(plan_args)
        test_goals[plan_num_blocks] = [generate_random_goal(world) for _ in range(num_goals)]

    # run planner for each model type and each random goal
    for method, method_model_paths in model_paths.items():
        success_data[method] = {}
        plan_paths[method] = {}
        for plan_num_blocks in all_plan_num_blocks:
            success_data[method][plan_num_blocks] = []
            plan_paths[method][plan_num_blocks] = []
            for model_path in method_model_paths:
                plan_args.model_exp_path = model_path
                model_logger = GoalConditionedExperimentLogger(model_path)
                model_args = model_logger.load_args()
                model_num_blocks = model_args.num_blocks
                plan_args.num_blocks = plan_num_blocks
                world = plan.setup_world(plan_args)
                all_successes = []
                for goal in test_goals[plan_num_blocks]:
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
                        all_successes.append(success)
                success_data[method][plan_num_blocks].append(np.sum(all_successes)/(num_goals*num_plan_attempts))
                plan_paths[method][plan_num_blocks].append(plan_exp_path)

    # plot colors
    cs = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    # plot all results
    figure, axis = plt.subplots()
    for i, (method, method_successes) in enumerate(success_data.items()):
        method_avgs = []
        method_mins = []
        method_maxs = []
        for plan_num_blocks, block_successes in method_successes.items():
            method_avgs.append(np.mean(success_data[method][plan_num_blocks]))
            method_mins.append(np.min(success_data[method][plan_num_blocks]))
            method_maxs.append(np.max(success_data[method][plan_num_blocks]))

        axis.plot(all_plan_num_blocks, method_avgs, color=cs[i], label=method)
        axis.fill_between(all_plan_num_blocks, method_mins, method_maxs, color=cs[i], alpha=0.1)

    axis.set_xticks(all_plan_num_blocks)
    axis.set_title('Planning Performance with Learned\nModels in 4 Block World')
    axis.set_xlabel('Number of Planning Blocks')
    axis.set_ylabel('% Success')
    axis.legend(title='Method')

    logger = GoalConditionedExperimentLogger.setup_experiment_directory(args, 'eval_goals')
    plt.savefig(logger.exp_path+'/eval_goals.png')
    logger.save_plot_data([plan_paths, success_data])
    print('Saving figures and data to %s.' % logger.exp_path)
    #except:
    #    import pdb; pdb.post_mortem()
