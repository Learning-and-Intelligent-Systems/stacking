import argparse
import numpy as np
import matplotlib.pyplot as plt

from tamp.predicates import On
from planning import plan
from learning.domains.abc_blocks.world import ABCBlocksWorldGT, ABCBlocksWorldGTOpt
from learning.active.utils import GoalConditionedExperimentLogger
from learning.domains.abc_blocks.abc_blocks_data import model_forward
from learning.evaluate.analyze_plans import vec_to_logical_state

def calc_accuracy(model_type, test_dataset, model=None, test_num_blocks=None):
    if model_type == 'learned':
        assert model is not None, 'model must be set to calc accuracy when model_type is learned'
        xs, ys = test_dataset[:]
        ys = ys.detach().numpy()
        preds = model_forward(model, xs)
        preds = preds.round()
        sum_axes = tuple(range(len(preds.shape)))[1:]
        accuracy = np.sum(np.all(ys == preds.round(), axis=sum_axes))/len(preds)
    elif model_type == 'opt':
        assert test_num_blocks is not None, 'test_num_blocks must be set to calc accuracy when model_type is opt'
        test_dataset.set_pred_type('full_state') # now ys will be full new edge states
        opt_world = ABCBlocksWorldGTOpt(test_num_blocks)
        preds = []
        for x, y in test_dataset:
            vof, vef, va = [xi.detach().numpy() for xi in x]
            lstate = vec_to_logical_state(vef, opt_world)
            new_state = opt_world.transition(lstate, va)
            v_new_edge_state = new_state.as_vec()[1]
            if np.array_equal(v_new_edge_state, y.detach().numpy()):
                preds.append(1)
            else:
                preds.append(0)
        accuracy = sum(preds)/len(preds)
    return accuracy

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
                        help='What type of search, mcts rollouts or a best-first with a learned heuristic (if args.eval_mode==planning)')
    parser.add_argument('--eval-mode',
                        type=str,
                        choices=['accuracy', 'planning'],
                        default='accuracy')
    parser.add_argument('--debug',
                        action='store_true',
                        help='set to run in debug mode')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    #### Parameters ####
     # used if args.eval_mode == 'planning'
    num_goals = 5
    num_plan_attempts = 1
    plan_args = argparse.Namespace(num_branches=10,
                        timeout=100,
                        c=.01,
                        max_ro=10,
                        search_mode=args.search_mode)

    # used if args.eval_mode == 'accuracy'

    # these are randomly generated datasets
    test_datasets = {2: 'learning/experiments/logs/datasets/large-test-2-20210810-223800',
                    3: 'learning/experiments/logs/datasets/large-test-3-20210810-223754',
                    4: 'learning/experiments/logs/datasets/large-test-4-20210810-223746',
                    5: 'learning/experiments/logs/datasets/large-test-5-20210810-223740',
                    6: 'learning/experiments/logs/datasets/large-test-6-20210810-223731'}
                    #7:''
                    #8:''}

    # used in both modes
    all_test_num_blocks = [2, 3, 4, 5, 6]#, 7, 8]     # NOTE: if args.eval_mode == 'accuracy', this must match test_datasets.keys()
    compare_opt = True                              # if want to compare against the optimistic model
    model_paths = {'FULL': ['learning/experiments/logs/models/4-block-full-20210811-094536',
                        'learning/experiments/logs/models/4-block-full-20210811-094558',
                        'learning/experiments/logs/models/4-block-full-20210811-094614',
                        'learning/experiments/logs/models/4-block-full-20210811-094627'],
                	'CLASS': ['learning/experiments/logs/models/fixed-opt-4-20210810-223512',
                        'learning/experiments/logs/models/fixed-opt-4-20210810-223533',
                        'learning/experiments/logs/models/fixed-opt-4-20210810-223551',
                        'learning/experiments/logs/models/fixed-opt-4-20210810-223610']}
    ########

    success_data = {}
    plan_paths = {}

    if args.eval_mode == 'planning':
        # generate random goals to evaluate planner with
        print('Generating random goals.')
        test_goals = {}
        for test_num_blocks in all_test_num_blocks:
            plan_args.num_blocks = test_num_blocks
            plan_args.model_exp_path = model_paths['FULL'][0]
            world = plan.setup_world(plan_args)
            test_goals[test_num_blocks] = [generate_random_goal(world) for _ in range(num_goals)]
        print('Done generating goals.')

    # run for each method and model
    for method, method_model_paths in model_paths.items():
        success_data[method] = {}
        plan_paths[method] = {}
        for test_num_blocks in all_test_num_blocks:
            success_data[method][test_num_blocks] = {}
            plan_paths[method][test_num_blocks] = {}
            for model_path in method_model_paths:
                model_logger = GoalConditionedExperimentLogger(model_path)
                model_args = model_logger.load_args()
                if args.eval_mode == 'planning':
                    plan_args.model_exp_path = model_path
                    plan_args.num_blocks = test_num_blocks
                    plan_args.model_type='learned'
                    world = plan.setup_world(plan_args)
                    all_successes = []
                    all_plan_exp_paths = []
                    for goal in test_goals[test_num_blocks]:
                        for _ in range(num_plan_attempts):
                            plan_args.exp_name = 'train_blocks_%i_policy_%s_plan_blocks_%i' % (model_args.num_blocks, model_args.policy, test_num_blocks)
                            found_plan, plan_exp_path = plan.run(goal, plan_args)
                            if found_plan:
                                test_world = ABCBlocksWorldGT(test_num_blocks)
                                final_state = test_world.execute_plan([node.action for node in found_plan])
                                success = test_world.is_goal_state(final_state, goal)
                            else:
                                success = False
                            #print_state(goal, world.num_objects)
                            all_successes.append(success)
                            all_plan_exp_paths.append(plan_exp_path)
                    success_data[method][test_num_blocks][model_path] = np.sum(all_successes)/(num_goals*num_plan_attempts)
                    plan_paths[method][test_num_blocks][model_path] = all_plan_exp_paths
                if args.eval_mode == 'accuracy':
                    trans_model = model_logger.load_trans_model()
                    #heur_model = model_logger.load_heur_model()
                    test_dataset_path = test_datasets[test_num_blocks]
                    test_dataset_logger = GoalConditionedExperimentLogger(test_dataset_path)
                    test_trans_dataset = test_dataset_logger.load_trans_dataset()
                    test_trans_dataset.set_pred_type(trans_model.pred_type)
                    #test_heur_dataset = test_dataset_logger.load_heur_dataset()
                    test_dataset_num_blocks = test_dataset_logger.args.num_blocks
                    assert test_dataset_num_blocks == test_num_blocks, 'Test dataset path %s does not contain %i blocks' % (test_dataset_path, test_num_blocks)
                    success_data[method][test_num_blocks][model_path] = calc_accuracy('learned', test_trans_dataset, model=trans_model)
                    #accuracies['heuristic'][train_num_blocks][0].append(test_dataset_num_blocks)
                    #accuracies['heuristic'][train_num_blocks][1].append(calc_accuracy(heur_model, test_heur_dataset))

    if compare_opt:
        success_data['OPT'] = {}
        for test_num_blocks in all_test_num_blocks:
            if args.eval_mode == 'planning':
                plan_args.num_blocks = test_num_blocks
                plan_args.model_type = 'opt'
                world = plan.setup_world(plan_args)
                all_successes = []
                all_plan_exp_paths = []
                for goal in test_goals[test_num_blocks]:
                    for _ in range(num_plan_attempts):
                        plan_args.exp_name = 'train_blocks_%i_policy_%s_plan_blocks_%i' % (model_args.num_blocks, model_args.policy, test_num_blocks)
                        found_plan, plan_exp_path = plan.run(goal, plan_args)
                        if found_plan:
                            test_world = ABCBlocksWorldGT(test_num_blocks)
                            final_state = test_world.execute_plan([node.action for node in found_plan])
                            success = test_world.is_goal_state(final_state, goal)
                        else:
                            success = False
                        #print_state(goal, world.num_objects)
                        all_successes.append(success)
                        all_plan_exp_paths.append(plan_exp_path)
                success_data['OPT'][test_num_blocks] = np.sum(all_successes)/(num_goals*num_plan_attempts)
                plan_paths['OPT'][test_num_blocks] = all_plan_exp_paths
            elif args.eval_mode == 'accuracy':
                test_dataset_path = test_datasets[test_num_blocks]
                test_dataset_logger = GoalConditionedExperimentLogger(test_dataset_path)
                test_trans_dataset = test_dataset_logger.load_trans_dataset()
                test_trans_dataset.set_pred_type(trans_model.pred_type)
                success_data['OPT'][test_num_blocks] = calc_accuracy('opt', test_trans_dataset, test_num_blocks=test_num_blocks)

    # Plot results

    # plot colors
    cs = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    # plot all results
    figure, axis = plt.subplots()
    for i, (method, method_successes) in enumerate(success_data.items()):
        method_avgs = []
        method_mins = []
        method_maxs = []
        for test_num_blocks, num_block_successes in method_successes.items():
            if method == 'OPT':
                num_block_success_data = num_block_successes
            else:
                num_block_success_data = [data for model_path, data in num_block_successes.items()]
            method_avgs.append(np.mean(num_block_success_data))
            method_mins.append(np.min(num_block_success_data))
            method_maxs.append(np.max(num_block_success_data))

        axis.plot(all_test_num_blocks, method_avgs, color=cs[i], label=method)
        axis.fill_between(all_test_num_blocks, method_mins, method_maxs, color=cs[i], alpha=0.1)

    axis.set_xticks(all_test_num_blocks)
    axis.set_title('Model Performance with Learned\nModels in %s Block World' % model_args.num_blocks) # TODO: hack
    axis.set_xlabel('Number of Planning Blocks')
    if args.eval_mode == 'planning':
        axis.set_ylabel('% Success')
    elif args.eval_mode == 'accuracy':
        axis.set_ylabel('Average Accuracy')
    axis.legend(title='Method')

    logger = GoalConditionedExperimentLogger.setup_experiment_directory(args, 'eval_models')
    plt.savefig(logger.exp_path+'/eval_models.png')
    logger.save_plot_data([plan_paths, success_data])
    print('Saving figures and data to %s.' % logger.exp_path)
    #except:
    #    import pdb; pdb.post_mortem()
