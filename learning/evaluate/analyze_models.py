import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

from tamp.predicates import On
from planning import plan
from learning.domains.abc_blocks.world import ABCBlocksWorldGT, ABCBlocksWorldGTOpt
from learning.active.utils import GoalConditionedExperimentLogger
from learning.domains.abc_blocks.abc_blocks_data import model_forward
from learning.evaluate.utils import vec_to_logical_state, plot_horiz_bars, join_strs, stacked_blocks_to_str

transition_names = ('state', 'action', 'next state', 'gt pred', \
                'model pred', 'optimistic next state', 'model accuracy')

def plot_accuracy(success_data, all_test_num_blocks, title, xlabel, ylabel, logger):
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
            method_mins.append(np.mean(num_block_success_data)-np.std(num_block_success_data))
            method_maxs.append(np.mean(num_block_success_data)+np.std(num_block_success_data))

        axis.plot(all_test_num_blocks, method_avgs, color=cs[i], label=method)
        axis.fill_between(all_test_num_blocks, method_mins, method_maxs, color=cs[i], alpha=0.1)

    axis.set_xticks(all_test_num_blocks)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.legend(title='Method')

    plt.savefig('%s/%s.png' % (logger.exp_path, title))
    print('Saving figures to %s.' % logger.exp_path)

def plot_transition_figs(transitions, model_logger, test_num_blocks, all_plot_inds, plot_keys):
    # The first value is how you want to separate the data
    # The second value is what you want to show on the bars

    for plot_key in plot_keys:
        y_axis_values = all_plot_inds[plot_key][0]
        bar_text_values = all_plot_inds[plot_key][1]
        y_label = join_strs(transition_names, y_axis_values)
        x_label = join_strs(transition_names, bar_text_values)
        color = False
        if plot_key == 'acc':
            color = True
        plot_horiz_bars(transitions, y_axis_values, bar_text_values, plot_title=plot_key, x_label=x_label, y_label=y_label, color=color)
        plot_path = '%s/%s_testblocks_%i.png' % (model_logger.exp_path, plot_key, test_num_blocks)
        plt.savefig(plot_path, bbox_inches = "tight")
        print('Saving accuracy plots to %s' % plot_path)
        plt.close()

def calc_trans_accuracy(model_type, test_dataset, test_num_blocks, model=None, return_transitions=False):
    '''
    :param model_type: in ['learned', 'opt']
    '''
    transitions = []
    accuracies = []

    gt_world = ABCBlocksWorldGT(test_num_blocks)
    opt_world = ABCBlocksWorldGTOpt(test_num_blocks)

    for x, y in test_dataset:
        # get all relevant transition info
        vof, vef, va = [xi.detach().numpy() for xi in x]
        action = [int(a) for a in va]
        lstate = vec_to_logical_state(vef, gt_world)
        lnext_state = gt_world.transition(lstate, action)
        lnext_opt_state = opt_world.transition(lstate, action)
        if model_type == 'opt':
            model_pred = 1. # optimistic model always thinks it's right
            gt_pred = np.array(np.array_equal(lnext_opt_state.as_vec()[1], lnext_state.as_vec()[1]))
        else:
            model_pred = model_forward(model, x).round().squeeze()
            gt_pred = y.numpy().squeeze()
        accuracy = int(np.array_equal(gt_pred, model_pred))# TODO: check this works in all model cases
        # turn all transition info into a string
        str_state = stacked_blocks_to_str(lstate.stacked_blocks)
        str_action = '%i/%i' % (action[0], action[1])
        str_next_state = stacked_blocks_to_str(lnext_state.stacked_blocks)
        if len(gt_pred.shape) > 1:
            str_gt_pred = stacked_blocks_to_str(vec_to_logical_state(gt_pred, gt_world).stacked_blocks)
            str_pred = stacked_blocks_to_str(vec_to_logical_state(model_pred, gt_world).stacked_blocks)
        else:
            str_gt_pred = '%i' % gt_pred # TODO: pred could be a state
            str_pred = '%i' % model_pred # TODO: pred could be a state
        str_opt_next_state = stacked_blocks_to_str(lnext_opt_state.stacked_blocks)
        str_acc = str(accuracy)

        transition = (str_state, str_action, str_next_state, str_gt_pred, \
                                str_pred, str_opt_next_state, str_acc)
        transitions.append(transition)
        accuracies.append(accuracy)

    final_accuracy = np.mean(accuracies)
    if return_transitions:
        return final_accuracy, transitions
    return final_accuracy

# Heuristic accuracy is not with respect to the training signal (which can be wrong
# due to random action selection) but with respect to the true minimum possible
# steps to goal
def calc_heur_error(model_type, test_dataset, test_num_blocks, model=None):
    '''
    :param model_type: in ['learned', 'opt']
    '''
    errors = []

    gt_world = ABCBlocksWorldGT(test_num_blocks)
    opt_world = ABCBlocksWorldGTOpt(test_num_blocks)

    for x, y in test_dataset:
        # get all relevant transition info
        vof, vef, vgef = [xi.detach().numpy() for xi in x]
        lstate = vec_to_logical_state(vef, gt_world)
        lgoal_state = vec_to_logical_state(vgef, gt_world)
        gt_pred = gt_world.steps_to_goal(lstate, lgoal_state)
        if model_type == 'opt':
            model_pred = opt_world.steps_to_goal(lstate, lgoal_state)
        else:
            model_pred = model_forward(model, x).round()
        error = np.linalg.norm(model_pred-gt_pred)
        errors.append(error)

    final_error = np.mean(errors)
    return errors

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
    ### used if args.eval_mode == 'planning' ###
    num_goals = 5
    num_plan_attempts = 1
    plan_args = argparse.Namespace(num_branches=10,
                        timeout=100,
                        c=.01,
                        max_ro=10,
                        search_mode=args.search_mode)

    ### used if args.eval_mode == 'accuracy' ###
    # these are randomly generated datasets
    test_datasets = {2: 'learning/experiments/logs/datasets/large-test-2-20210810-223800',
                    3: 'learning/experiments/logs/datasets/large-test-3-20210810-223754',
                    4: 'learning/experiments/logs/datasets/large-test-4-20210810-223746',
                    #5: 'learning/experiments/logs/datasets/large-test-5-20210810-223740',
                    #6: 'learning/experiments/logs/datasets/large-test-6-20210810-223731',
                    #7:'learning/experiments/logs/datasets/large-test-7-20210811-173148',
                    #8:'learning/experiments/logs/datasets/large-test-8-20210811-173210'}
                    }
    # name of plot to generate: (how to fivide data on y axis, what to show on x axis)
    all_plot_inds = {'all_trans': [[0,1,2,3],[4]],
                'classification': [[3], [0,1,2]],
                'init_state': [[0,3], [1,2]],
                'acc': [[0,1], [6]]}
    # which ones to actually plot and save to logger
    plot_keys = []#['acc']

    ### used in both modes ###
    all_test_num_blocks = [2, 3, 4]#, 8]#[2, 3, 4, 5, 6, 7, 8]     # NOTE: if args.eval_mode == 'accuracy', this must match test_datasets.keys()
    compare_opt = True                              # if want to compare against the optimistic model
    model_paths = {'FULL': ['learning/experiments/logs/models/delta-4-block-random-20210811-171313',
                                'learning/experiments/logs/models/delta-4-block-random-20210811-171322',
                                'learning/experiments/logs/models/delta-4-block-random-20210811-171330',
                                #'learning/experiments/logs/models/delta-4-block-random-20210811-171339',
                                #'learning/experiments/logs/models/delta-4-block-random-20210811-171347',
                                #'learning/experiments/logs/models/delta-4-block-random-20210811-171356',
                                #'learning/experiments/logs/models/delta-4-block-random-20210811-171404',
                                #'learning/experiments/logs/models/delta-4-block-random-20210811-171413',
                                #'learning/experiments/logs/models/delta-4-block-random-20210811-171421',
                                #'learning/experiments/logs/models/delta-4-block-random-20210811-171430',
                                #'learning/experiments/logs/models/delta-4-block-random-20210811-171438',
                                #'learning/experiments/logs/models/delta-4-block-random-20210811-171447',
                                #'learning/experiments/logs/models/delta-4-block-random-20210811-171456',
                                #'learning/experiments/logs/models/delta-4-block-random-20210811-171504',
                                #'learning/experiments/logs/models/delta-4-block-random-20210811-171513',
                                #'learning/experiments/logs/models/delta-4-block-random-20210811-171522',
                                #'learning/experiments/logs/models/delta-4-block-random-20210811-171530',
                                #'learning/experiments/logs/models/delta-4-block-random-20210811-171539',
                                #'learning/experiments/logs/models/delta-4-block-random-20210811-171548',
                                'learning/experiments/logs/models/delta-4-block-random-20210811-171557'],
                	'CLASS': ['learning/experiments/logs/models/class-4-block-random-20210811-170709',
                                'learning/experiments/logs/models/class-4-block-random-20210811-170858',
                                'learning/experiments/logs/models/class-4-block-random-20210811-170719',
                                #'learning/experiments/logs/models/class-4-block-random-20210811-170909',
                                #'learning/experiments/logs/models/class-4-block-random-20210811-170730',
                                #'learning/experiments/logs/models/class-4-block-random-20210811-170920',
                                #'learning/experiments/logs/models/class-4-block-random-20210811-170740',
                                #'learning/experiments/logs/models/class-4-block-random-20210811-170936',
                                #'learning/experiments/logs/models/class-4-block-random-20210811-170750',
                                #'learning/experiments/logs/models/class-4-block-random-20210811-171043',
                                #'learning/experiments/logs/models/class-4-block-random-20210811-170802',
                                #'learning/experiments/logs/models/class-4-block-random-20210811-171054',
                                #'learning/experiments/logs/models/class-4-block-random-20210811-170814',
                                #'learning/experiments/logs/models/class-4-block-random-20210811-171125',
                                #'learning/experiments/logs/models/class-4-block-random-20210811-170825',
                                #'learning/experiments/logs/models/class-4-block-random-20210811-171142',
                                #'learning/experiments/logs/models/class-4-block-random-20210811-170835',
                                #'learning/experiments/logs/models/class-4-block-random-20210811-171210',
                                #'learning/experiments/logs/models/class-4-block-random-20210811-170847',
                                'learning/experiments/logs/models/class-4-block-random-20210811-171218']}

########

    trans_success_data = {}
    heur_success_data = {}
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
        trans_success_data[method] = {}
        heur_success_data[method] = {}
        plan_paths[method] = {}
        for test_num_blocks in all_test_num_blocks:
            trans_success_data[method][test_num_blocks] = {}
            heur_success_data[method][test_num_blocks] = {}
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
                    trans_success_data[method][test_num_blocks][model_path] = np.sum(all_successes)/(num_goals*num_plan_attempts)
                    plan_paths[method][test_num_blocks][model_path] = all_plan_exp_paths
                if args.eval_mode == 'accuracy':
                    trans_model = model_logger.load_trans_model()
                    heur_model = model_logger.load_heur_model()
                    test_dataset_path = test_datasets[test_num_blocks]
                    test_dataset_logger = GoalConditionedExperimentLogger(test_dataset_path)
                    test_trans_dataset = test_dataset_logger.load_trans_dataset()
                    test_trans_dataset.set_pred_type(trans_model.pred_type)
                    test_heur_dataset = test_dataset_logger.load_heur_dataset()
                    assert test_dataset_logger.args.num_blocks == test_num_blocks, \
                            'Test dataset path %s does not contain %i blocks' % \
                            (test_dataset_path, test_num_blocks)
                    trans_accuracy, transitions = calc_trans_accuracy('learned', \
                                                    test_trans_dataset, test_num_blocks, \
                                                    model=trans_model, return_transitions=True)
                    heur_accuracy = calc_heur_error('learned', test_heur_dataset, \
                                                        test_num_blocks, model=heur_model)
                    trans_success_data[method][test_num_blocks][model_path] = trans_accuracy
                    heur_success_data[method][test_num_blocks][model_path] = heur_accuracy
                    plot_transition_figs(transitions, model_logger, test_num_blocks, \
                                        all_plot_inds, plot_keys)

    if compare_opt:
        trans_success_data['OPT'] = {}
        heur_success_data['OPT'] = {}
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
                        all_successes.append(success)
                        all_plan_exp_paths.append(plan_exp_path)
                trans_success_data['OPT'][test_num_blocks] = np.sum(all_successes)/(num_goals*num_plan_attempts)
                plan_paths['OPT'][test_num_blocks] = all_plan_exp_paths
            elif args.eval_mode == 'accuracy':
                test_dataset_path = test_datasets[test_num_blocks]
                test_dataset_logger = GoalConditionedExperimentLogger(test_dataset_path)
                test_trans_dataset = test_dataset_logger.load_trans_dataset()
                test_heur_dataset = test_dataset_logger.load_heur_dataset()
                test_trans_dataset.set_pred_type('full_state')
                trans_success_data['OPT'][test_num_blocks] = calc_trans_accuracy('opt', test_trans_dataset, test_num_blocks)
                heur_success_data['OPT'][test_num_blocks] = calc_heur_error('opt', test_heur_dataset, test_num_blocks)

    # Save data to logger
    logger = GoalConditionedExperimentLogger.setup_experiment_directory(args, 'eval_models')
    logger.save_plot_data([plan_paths, trans_success_data, heur_success_data])
    print('Saving data to %s.' % logger.exp_path)

    # Plot results and save to logger
    xlabel = 'Number of Test Blocks'

    trans_title = 'Transition Model Performance with Learned\nModels in %s Block World' % model_args.num_blocks  # TODO: hack
    trans_ylabel = '% Success' if args.eval_mode == 'planning' else 'Average Accuracy'
    plot_accuracy(trans_success_data, all_test_num_blocks, trans_title, xlabel, trans_ylabel, logger)

    if args.eval_mode == 'accuracy':
        heur_title = 'Heuristic Model Performance with Learned\nModels in %s Block World' % model_args.num_blocks  # TODO: hack
        heur_ylabel = 'Average MSE'
        plot_accuracy(heur_success_data, all_test_num_blocks, heur_title, xlabel, heur_ylabel, logger)
