import argparse
import numpy as np
import matplotlib.pyplot as plt

from tamp.predicates import On
from planning import plan
from learning.domains.abc_blocks.world import ABCBlocksWorldGT, generate_random_goal
from learning.active.utils import GoalConditionedExperimentLogger
from learning.domains.abc_blocks.abc_blocks_data import model_forward
from learning.evaluate.utils import plot_results, recc_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name',
                        type=str,
                        required=True,
                        help='where to save exp data')
    parser.add_argument('--debug',
                        action='store_true',
                        help='set to run in debug mode')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    #### Parameters ####
    num_goals = 5
    num_plan_attempts = 1
    plan_args = argparse.Namespace(num_branches=10,
                        timeout=100,
                        c=.01,
                        max_ro=10)
    all_test_num_blocks = [2, 3, 4,5, 6, 7, 8]
    full_model_paths = ['learning/experiments/logs/models/delta-4-block-random-20210811-171313',
                        'learning/experiments/logs/models/delta-4-block-random-20210811-171322',
                        'learning/experiments/logs/models/delta-4-block-random-20210811-171330',
                        'learning/experiments/logs/models/delta-4-block-random-20210811-171339',
                        'learning/experiments/logs/models/delta-4-block-random-20210811-171347',
                        'learning/experiments/logs/models/delta-4-block-random-20210811-171356',
                        'learning/experiments/logs/models/delta-4-block-random-20210811-171404',
                        'learning/experiments/logs/models/delta-4-block-random-20210811-171413',
                        'learning/experiments/logs/models/delta-4-block-random-20210811-171421',
                        'learning/experiments/logs/models/delta-4-block-random-20210811-171430',
                        'learning/experiments/logs/models/delta-4-block-random-20210811-171438',
                        'learning/experiments/logs/models/delta-4-block-random-20210811-171447',
                        'learning/experiments/logs/models/delta-4-block-random-20210811-171456',
                        'learning/experiments/logs/models/delta-4-block-random-20210811-171504',
                        'learning/experiments/logs/models/delta-4-block-random-20210811-171513',
                        'learning/experiments/logs/models/delta-4-block-random-20210811-171522',
                        'learning/experiments/logs/models/delta-4-block-random-20210811-171530',
                        'learning/experiments/logs/models/delta-4-block-random-20210811-171539',
                        'learning/experiments/logs/models/delta-4-block-random-20210811-171548',
                        'learning/experiments/logs/models/delta-4-block-random-20210811-171557']
    class_model_paths = ['learning/experiments/logs/models/class-4-block-random-20210811-170709',
                        'learning/experiments/logs/models/class-4-block-random-20210811-170858',
                        'learning/experiments/logs/models/class-4-block-random-20210811-170719',
                        'learning/experiments/logs/models/class-4-block-random-20210811-170909',
                        'learning/experiments/logs/models/class-4-block-random-20210811-170730',
                        'learning/experiments/logs/models/class-4-block-random-20210811-170920',
                        'learning/experiments/logs/models/class-4-block-random-20210811-170740',
                        'learning/experiments/logs/models/class-4-block-random-20210811-170936',
                        'learning/experiments/logs/models/class-4-block-random-20210811-170750',
                        'learning/experiments/logs/models/class-4-block-random-20210811-171043',
                        'learning/experiments/logs/models/class-4-block-random-20210811-170802',
                        'learning/experiments/logs/models/class-4-block-random-20210811-171054',
                        'learning/experiments/logs/models/class-4-block-random-20210811-170814',
                        'learning/experiments/logs/models/class-4-block-random-20210811-171125',
                        'learning/experiments/logs/models/class-4-block-random-20210811-170825',
                        'learning/experiments/logs/models/class-4-block-random-20210811-171142',
                        'learning/experiments/logs/models/class-4-block-random-20210811-170835',
                        'learning/experiments/logs/models/class-4-block-random-20210811-171210',
                        'learning/experiments/logs/models/class-4-block-random-20210811-170847',
                        'learning/experiments/logs/models/class-4-block-random-20210811-171218']
    compare_methods = {#'FULL': {'model_type': 'learned',
                        #        'value_fns': ['rollout', 'learned'],
                        #        'model_paths': full_model_paths},
                        'CLASS': {'model_type': 'learned',
                                'value_fns': ['rollout', 'learned'],
                                'model_paths': class_model_paths},
                        'TRUE': {'model_type': 'true',
                                'value_fns': ['rollout'],
                                'model_paths': [None]},
                        'OPT': {'model_type': 'opt',
                                'value_fns': ['rollout'],
                                'model_paths': [None]}}
########

    success_data = recc_dict()
    rank_success_data = recc_dict()
    plan_paths = recc_dict()

    # generate random goals to evaluate planner with
    print('Generating random goals.')
    test_goals = {}
    for test_num_blocks in all_test_num_blocks:
        world = ABCBlocksWorldGT(test_num_blocks)
        test_goals[test_num_blocks] = [generate_random_goal(world) for _ in range(num_goals)]
    print('Done generating goals.')

    for method_name in compare_methods:
        model_type = compare_methods[method_name]['model_type']
        plan_args.model_type = model_type
        for test_num_blocks in all_test_num_blocks:
            plan_args.num_blocks = test_num_blocks
            for model_path in compare_methods[method_name]['model_paths']:
                if model_path is not None:
                    model_logger = GoalConditionedExperimentLogger(model_path)
                    model_args = model_logger.load_args()
                    plan_args.model_exp_path = model_path
                for value_fn in compare_methods[method_name]['value_fns']:
                    plan_args.value_fn = value_fn
                    all_successes, all_plan_exp_paths, all_rank_accuracies = [], [], []
                    for goal in test_goals[test_num_blocks]:
                        for _ in range(num_plan_attempts):
                            plan_args.exp_name = 'train_blocks_%i_policy_%s_plan_blocks_%i' % (model_args.num_blocks, model_args.policy, test_num_blocks)
                            found_plan, plan_exp_path, rank_accuracy = plan.run(goal, plan_args)
                            if found_plan:
                                test_world = ABCBlocksWorldGT(test_num_blocks)
                                trajectory = test_world.execute_plan(found_plan)
                                final_state = trajectory[-1][0]
                                success = test_world.is_goal_state(final_state, goal)
                            else:
                                success = False
                            all_successes.append(success)
                            all_plan_exp_paths.append(plan_exp_path)
                            all_rank_accuracies.append(rank_accuracy)
                    full_method = '%s-%s' % (method_name, value_fn)
                    rank_success_data[full_method][test_num_blocks][model_path] = np.mean(all_rank_accuracies)
                    success_data[full_method][test_num_blocks][model_path] = np.mean(all_successes)
                    plan_paths[full_method][test_num_blocks][model_path] = all_plan_exp_paths

    # Save data to logger
    logger = GoalConditionedExperimentLogger.setup_experiment_directory(args, 'plan_results')
    logger.save_plot_data([plan_paths, success_data, rank_success_data])
    print('Saving data to %s.' % logger.exp_path)

    # Plot results and save to logger
    xlabel = 'Number of Test Blocks'
    trans_title = 'Planning Performance with Learned\nModels in %s Block World' % model_args.num_blocks  # TODO: hack (assumes only one train num blocks)
    trans_ylabel = '% Success'
    plot_results(success_data, all_test_num_blocks, trans_title, xlabel, trans_ylabel, logger)

    heur_title = 'Heuristic Model Rank Accuracy with Learned\nModels in %s Block World' % model_args.num_blocks  # TODO: hack (assumes only one train num blocks)
    heur_ylabel = 'Rank Accuracy'
    plot_results(rank_success_data, all_test_num_blocks, heur_title, xlabel, heur_ylabel, logger)
