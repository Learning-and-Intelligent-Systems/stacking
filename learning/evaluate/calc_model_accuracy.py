import argparse
import numpy as np
import matplotlib.pyplot as plt

from tamp.predicates import On
from planning import plan
from learning.domains.abc_blocks.world import ABCBlocksWorldGT, ABCBlocksWorldGTOpt
from learning.active.utils import GoalConditionedExperimentLogger
from learning.domains.abc_blocks.abc_blocks_data import model_forward
from learning.evaluate.utils import vec_to_logical_state, plot_horiz_bars, join_strs, \
                                stacked_blocks_to_str, plot_results, recc_dict, potential_actions

def calc_full_trans_accuracy(model_type, test_num_blocks, model):
    '''
    :param model_type: in ['learned', 'opt']
    '''
    pos_actions, neg_actions = potential_actions(test_num_blocks)
    accuracies = {}
    # NOTE: we test all actions from initial state assuming that the network is ignoring the state
    world = ABCBlocksWorldGT(test_num_blocks)
    init_state = world.get_init_state()
    vof, vef = init_state.as_vec()
    for gt_pred, actions in zip([1, 0], [pos_actions, neg_actions]):
        for action in actions:
            if model_type == 'opt':
                model_pred = 1. # optimistic model always thinks it's right
            else:
                int_action = [int(action[0]), int(action[-1])]
                model_pred = model_forward(model, [vof, vef, int_action]).round().squeeze()
            accuracies[(gt_pred, action)] = int(np.array_equal(gt_pred, model_pred))
    return accuracies

def plot_full_accuracies(method, method_full_trans_success_data):
    fig, axes = plt.subplots(len(method_full_trans_success_data), 1, sharex=True)

    def plot_accs(avg_acc, std_acc, action, nb, ni):
        axes[ni].bar(action,
                    avg_acc,
                    color='r' if avg_acc<0.5 else 'g',
                    yerr=std_acc)
        axes[ni].set_ylabel('Num Blocks = %i' % nb)

    for ni, (num_blocks, num_blocks_data) in enumerate(method_full_trans_success_data.items()):
        try:
            all_models = list(num_blocks_data.keys())
            for (label, action), accs in num_blocks_data[all_models[0]].items(): # NOTE: all models have same keys
                all_accs = [num_blocks_data[model][(label, action)] for model in all_models]
                avg_acc = np.mean(all_accs)
                std_acc = np.std(all_accs)
                plot_accs(avg_acc, std_acc, action, num_blocks, ni)
        except: # model is opt, not learned
            for (action, label), accs in num_blocks_data:
                avg_acc = accs
                std_acc = 0
                plot_accs(avg_acc, std_acc, action, num_blocks, ni)

    fig.suptitle('Accuracy of Method %s on Different Test Set Num Blocks' % method)
    plt.xlabel('Actions')
    #plt.show()
    plt.savefig('all_transition_accuracies_method_%s' % method )
    #plt.close()

def plot_transition_figs(transitions, model_logger, test_num_blocks, all_plot_inds, \
                        plot_keys, transition_names):
    # The first value is how you want to separate the data
    # The second value is what you want to show on the bars

    for plot_key in plot_keys:
        y_axis_values = all_plot_inds[plot_key][0]
        bar_text_values = all_plot_inds[plot_key][1]
        y_label = join_strs(transition_names, y_axis_values)
        x_label = join_strs(transition_names, bar_text_values)
        color = False
        if plot_key in ['acc', 'top on table', 'plus one', 'bottom pos', 'opt+bottom_pos', 'opt+plus_one']:
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

        # calculate state and action features
        bottom_num = action[0]
        top_num = action[1]
        plus_one = str(int(top_num == bottom_num + 1))
        # NOTE: these are the only 2 cases. we don't sameple actions where
        # the bottom block is in the middle or bottom of the stack
        if len(lstate.stacked_blocks) == 0:
            bottom_pos = '0' # no stack
        elif lstate.stacked_blocks[-1] == bottom_num:
            bottom_pos = '1' # top of stack

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
                                str_pred, str_opt_next_state, str_acc, \
                                plus_one, bottom_pos)
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
    # these datasets are generated with random exploration
    test_datasets = {2: 'learning/experiments/logs/datasets/large-test-2-20210810-223800',
                    3: 'learning/experiments/logs/datasets/large-test-3-20210810-223754',
                    4: 'learning/experiments/logs/datasets/large-test-4-20210810-223746',
                    5: 'learning/experiments/logs/datasets/large-test-5-20210810-223740',
                    6: 'learning/experiments/logs/datasets/large-test-6-20210810-223731',
                    7:'learning/experiments/logs/datasets/large-test-7-20210811-173148',
                    8:'learning/experiments/logs/datasets/large-test-8-20210811-173210'}

    # name of plot to generate: (how to divide data on y axis, what to show on x axis)
    # indices are from transition_names at top of file
    transition_names = ('state', 'action', 'next state', 'gt pred', \
                    'model pred', 'optimistic next state', 'model accuracy', \
                    'plus one', 'bottom pos')
    all_plot_inds = {'plus one': [[7, 6], [6]],
                     'bottom pos': [[8, 6], [6]],
                     'all_trans': [[0,1,2,3],[4]],
                     'classification': [[3], [0,1,2]],
                     'init_state': [[0,3], [1,2]],
                     'acc': [[0,1], [6]],
                     'opt': [[3, 4, 6], [6]],
                     'opt+bottom_pos': [[3, 4, 6, 8], [6]],
                     'opt+plus_one': [[3, 4, 6,7], [6]]}
    # which ones to actually plot and save to logger
    plot_keys = ['opt+bottom_pos', 'opt+plus_one']#['plus one', 'bottom pos'] #, 'acc']
    compare_opt = False  # if want to compare against the optimistic model

    deep = ['learning/experiments/logs/models/deep-20210823-143508',
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
    exploit_T_opt_deep = ['learning/experiments/logs/models/model-test-random-goals-opt-20210823-165549',
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
    exploit_T_learned_deep = ['learning/experiments/logs/models/model-test-random-goals-learned-20210823-180152',
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

    model_paths = {'deep': deep,
                    'exploit-T-opt-deep': exploit_T_opt_deep,
                    'exploit-T-learned-deep': exploit_T_learned_deep}
########

    trans_success_data = recc_dict()
    full_trans_success_data = recc_dict()
    heur_success_data = recc_dict()

    # run for each method and model
    for method, method_model_paths in model_paths.items():
        for test_num_blocks in test_datasets:
            for model_path in method_model_paths:
                model_logger = GoalConditionedExperimentLogger(model_path)
                model_args = model_logger.load_args()
                trans_model = model_logger.load_trans_model()
                #heur_model = model_logger.load_heur_model()
                test_dataset_path = test_datasets[test_num_blocks]
                test_dataset_logger = GoalConditionedExperimentLogger(test_dataset_path)
                test_trans_dataset = test_dataset_logger.load_trans_dataset()
                test_trans_dataset.set_pred_type(trans_model.pred_type)
                #test_heur_dataset = test_dataset_logger.load_heur_dataset()
                #assert test_dataset_logger.args.num_blocks == test_num_blocks, \
                #        'Test dataset path %s does not contain %i blocks' % \
                #        (test_dataset_path, test_num_blocks)
                trans_accuracy, transitions = calc_trans_accuracy('learned', \
                                                test_trans_dataset, test_num_blocks, \
                                                model=trans_model, return_transitions=True)
                full_trans_success_data[method][test_num_blocks][model_path] = \
                                        calc_full_trans_accuracy('learned', \
                                                                test_num_blocks, \
                                                                model=trans_model)
                #heur_accuracy = calc_heur_error('learned', test_heur_dataset, \
                #                                    test_num_blocks, model=heur_model)
                trans_success_data[method][test_num_blocks][model_path] = trans_accuracy
                #heur_success_data[method][test_num_blocks][model_path] = heur_accuracy
                #plot_transition_figs(transitions, model_logger, test_num_blocks, \
                #                    all_plot_inds, plot_keys, transition_names)

    if compare_opt:
        for test_num_blocks in test_datasets:
            test_dataset_path = test_datasets[test_num_blocks]
            test_dataset_logger = GoalConditionedExperimentLogger(test_dataset_path)
            test_trans_dataset = test_dataset_logger.load_trans_dataset()
            test_heur_dataset = test_dataset_logger.load_heur_dataset()
            test_trans_dataset.set_pred_type('full_state')
            trans_success_data['OPT'][test_num_blocks] = calc_trans_accuracy('opt', test_trans_dataset, test_num_blocks)
            full_trans_success_data['OPT'][test_num_blocks] = calc_full_trans_accuracy('opt', \
                                                            test_num_blocks, \
                                                            model=trans_model)
            #heur_accuracy = calc_heur_error('opt', test_heur_dataset, test_num_blocks)
            #heur_success_data['OPT'][test_num_blocks] = heur_accuracy

    # Save data to logger
    logger = GoalConditionedExperimentLogger.setup_experiment_directory(args, 'model_accuracy')
    logger.save_plot_data([test_datasets, trans_success_data, heur_success_data])
    print('Saving data to %s.' % logger.exp_path)

    # Plot results and save to logger
    xlabel = 'Number of Test Blocks'
    trans_title = 'Transition Model Performance with Learned\nModels in %s Block World' % model_args.num_blocks  # TODO: hack
    trans_ylabel = 'Average Accuracy'
    all_test_num_blocks = list(test_datasets.keys())
    plot_results(trans_success_data, all_test_num_blocks, trans_title, xlabel, trans_ylabel, logger)

    for method, method_data in full_trans_success_data.items():
        plot_full_accuracies(method, method_data)
    plt.show()
    #heur_title = 'Heuristic Model MSE with Learned\nModels in %s Block World' % model_args.num_blocks  # TODO: hack
    #heur_ylabel = 'Average MSE'
    #plot_results(heur_success_data, all_test_num_blocks, heur_title, xlabel, heur_ylabel, logger)
