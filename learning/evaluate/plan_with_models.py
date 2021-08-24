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
    num_goals = 10
    base_plan_args = argparse.Namespace(num_branches=10,
                        timeout=100,
                        c=.01,
                        max_ro=10)
    all_test_num_blocks = [2, 3, 4, 5, 6, 7, 8]
    random_actions = ['learning/experiments/logs/models/deep-20210823-143508',
                        'learning/experiments/logs/models/deep-20210823-143510',
                        'learning/experiments/logs/models/deep-20210823-143512',
                        'learning/experiments/logs/models/deep-20210823-143514',
                        'learning/experiments/logs/models/deep-20210823-143516',
                        'learning/experiments/logs/models/deep-20210823-143518',
                        'learning/experiments/logs/models/deep-20210823-143520',
                        'learning/experiments/logs/models/deep-20210823-143522',
                        'learning/experiments/logs/models/deep-20210823-143523',
                        'learning/experiments/logs/models/deep-20210823-143525',
                        'learning/experiments/logs/models/deep-20210823-143527',
                        'learning/experiments/logs/models/deep-20210823-143529',
                        'learning/experiments/logs/models/deep-20210823-143531',
                        'learning/experiments/logs/models/deep-20210823-143533',
                        'learning/experiments/logs/models/deep-20210823-143535',
                        'learning/experiments/logs/models/deep-20210823-143537',
                        'learning/experiments/logs/models/deep-20210823-143539',
                        'learning/experiments/logs/models/deep-20210823-143541',
                        'learning/experiments/logs/models/deep-20210823-143543',
                        'learning/experiments/logs/models/deep-20210823-143545']
    random_goals_opt = ['learning/experiments/logs/models/model-test-random-goals-opt-20210823-165549',
                        'learning/experiments/logs/models/model-test-random-goals-opt-20210823-165810',
                        'learning/experiments/logs/models/model-test-random-goals-opt-20210823-170034',
                        'learning/experiments/logs/models/model-test-random-goals-opt-20210823-170322',
                        'learning/experiments/logs/models/model-test-random-goals-opt-20210823-170613',
                        'learning/experiments/logs/models/model-test-random-goals-opt-20210823-171415',
                        'learning/experiments/logs/models/model-test-random-goals-opt-20210823-174213',
                        'learning/experiments/logs/models/model-test-random-goals-opt-20210823-174438',
                        'learning/experiments/logs/models/model-test-random-goals-opt-20210823-174637',
                        'learning/experiments/logs/models/model-test-random-goals-opt-20210823-174845',
                        'learning/experiments/logs/models/model-test-random-goals-opt-20210823-175114',
                        'learning/experiments/logs/models/model-test-random-goals-opt-20210823-175401',
                        'learning/experiments/logs/models/model-test-random-goals-opt-20210823-175731',
                        'learning/experiments/logs/models/model-test-random-goals-opt-20210823-175921',
                        'learning/experiments/logs/models/model-test-random-goals-opt-20210823-180142',
                        'learning/experiments/logs/models/model-test-random-goals-opt-20210823-180404',
                        'learning/experiments/logs/models/model-test-random-goals-opt-20210823-180615',
                        'learning/experiments/logs/models/model-test-random-goals-opt-20210823-180734',
                        'learning/experiments/logs/models/model-test-random-goals-opt-20210823-181009',
                        'learning/experiments/logs/models/model-test-random-goals-opt-20210823-181216']
    random_goals_learned = ['learning/experiments/logs/models/model-test-random-goals-learned-20210823-180152',
                        'learning/experiments/logs/models/model-test-random-goals-learned-20210823-165618',
                        'learning/experiments/logs/models/model-test-random-goals-learned-20210823-170031',
                        'learning/experiments/logs/models/model-test-random-goals-learned-20210823-170422',
                        'learning/experiments/logs/models/model-test-random-goals-learned-20210823-171341',
                        'learning/experiments/logs/models/model-test-random-goals-learned-20210823-174239',
                        'learning/experiments/logs/models/model-test-random-goals-learned-20210823-174658',
                        'learning/experiments/logs/models/model-test-random-goals-learned-20210823-175109',
                        'learning/experiments/logs/models/model-test-random-goals-learned-20210823-175726',
                        'learning/experiments/logs/models/model-test-random-goals-learned-20210823-180630',
                        'learning/experiments/logs/models/model-test-random-goals-learned-20210823-181136',
                        'learning/experiments/logs/models/model-test-random-goals-learned-20210823-181550',
                        'learning/experiments/logs/models/model-test-random-goals-learned-20210823-181840',
                        'learning/experiments/logs/models/model-test-random-goals-learned-20210823-183900',
                        'learning/experiments/logs/models/model-test-random-goals-learned-20210823-185941',
                        'learning/experiments/logs/models/model-test-random-goals-learned-20210823-190252',
                        'learning/experiments/logs/models/model-test-random-goals-learned-20210823-190613',
                        'learning/experiments/logs/models/model-test-random-goals-learned-20210823-191022',
                        'learning/experiments/logs/models/model-test-random-goals-learned-20210823-191356',
                        'learning/experiments/logs/models/model-test-random-goals-learned-20210823-191700']
    compare_methods = {'T-opt':                {'model_type': 'opt',
                                                'value_fns': ['rollout'],
                                                'model_paths': [None]},
                        'explore':             {'model_type': 'learned',
                                                'value_fns': ['rollout'],
                                                'model_paths': random_actions},
                        'exploit-T-opt':        {'model_type': 'learned',
                                                'value_fns': ['rollout'],
                                                'model_paths': random_goals_opt},
                        'exploit-T-learned':    {'model_type': 'learned',
                                                'value_fns': ['rollout'],
                                                'model_paths': random_goals_learned}}
    
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
        plan_args = base_plan_args
        plan_args.model_type = model_type
        for test_num_blocks in all_test_num_blocks:
            plan_args.num_blocks = test_num_blocks
            for model_path in compare_methods[method_name]['model_paths']:
                if model_path is not None:
                    model_logger = GoalConditionedExperimentLogger(model_path)
                    model_args = model_logger.load_args()
                    plan_args.model_exp_path = model_path
                    plan_args.exp_name = 'train_blocks_%i_plan_blocks_%i' % (model_args.num_blocks, test_num_blocks)
                else:
                    plan_args.exp_name = 'plan_blocks_%i' % (test_num_blocks)
                for value_fn in compare_methods[method_name]['value_fns']:
                    plan_args.value_fn = value_fn
                    all_successes, all_plan_exp_paths, all_rank_accuracies = [], [], []
                    for goal in test_goals[test_num_blocks]:
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
                    full_method = '%s' % method_name
                    rank_success_data[full_method][test_num_blocks][model_path] = np.mean(all_rank_accuracies)
                    success_data[full_method][test_num_blocks][model_path] = np.mean(all_successes)
                    plan_paths[full_method][test_num_blocks][model_path] = all_plan_exp_paths

    # Save data to logger
    logger = GoalConditionedExperimentLogger.setup_experiment_directory(args, 'plan_results')
    logger.save_plot_data([plan_paths, success_data, rank_success_data])
    print('Saving data to %s.' % logger.exp_path)

    # Plot results and save to logger
    xlabel = 'Number of Test Blocks'
    trans_title = 'Planning Performance'   # TODO: hack (assumes only one train num blocks)
    try:
        trans_title += ' with Learned\nModels in %s Block World' % model_args.num_blocks
    except:
        pass
    trans_ylabel = '% Success'
    plot_results(success_data, all_test_num_blocks, trans_title, xlabel, trans_ylabel, logger)

    heur_title = 'Heuristic Model'  # TODO: hack (assumes only one train num blocks)
    try:
        trans_title += ' with Learned\nModels in %s Block World' % model_args.num_blocks
    except:
        pass
    heur_ylabel = 'Rank Accuracy'
    plot_results(rank_success_data, all_test_num_blocks, heur_title, xlabel, heur_ylabel, logger)
