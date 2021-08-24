from operator import itemgetter
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from learning.domains.abc_blocks.world import LogicalState

# this generally doesn't work, but works when the pred_type == class
# because the edge states are all valid states
def vec_to_logical_state(edge_features, world):
    state = LogicalState(world._blocks, world.num_objects, world.table)
    state.stacked_blocks = vec_state_to_stacked_blocks(edge_features)
    return state

def vec_state_to_stacked_blocks(edge_features):
    stacked_blocks = set()
    num_objects = edge_features.shape[0]
    for bottom_i in range(num_objects):
        for top_i in range(num_objects):
            if edge_features[bottom_i, top_i] == 1.:
                if bottom_i != 0 and top_i != 0: # 0 is the table
                    stacked_blocks.add(bottom_i)
                    stacked_blocks.add(top_i)
    stacked_blocks = list(stacked_blocks)
    stacked_blocks.sort()
    return stacked_blocks

def stacked_blocks_to_str(stacked_blocks):
    if len(stacked_blocks) > 0:
        str_state = ''
        str_state += str(stacked_blocks[0])
        for block_id in stacked_blocks[1:]:
            str_state += '/'+str(block_id)
    else:
        str_state = '-'
    return str_state


def join_strs(str_list, indices):
    if len(indices) == 1:
        return str_list[indices[0]]
    else:
        return ' , '.join(itemgetter(*indices)(str_list))

def plot_horiz_bars(transitions, plot_inds, bar_text_inds, color=False, plot_title=None, y_label=None, x_label=None):

    plot_data = {}
    plot_text = {}
    for transition in transitions:
        key = join_strs(transition, plot_inds)
        bar_text = join_strs(transition, bar_text_inds)
        if key in plot_data:
            plot_data[key] += 1
            plot_text[key].add(bar_text)
        else:
            plot_data[key] = 1
            plot_text[key] = set()
            plot_text[key].add(bar_text)

    fig, ax = plt.subplots()
    if color:
        # this colors the bar based off of the first value in the bar text (i think)
        bar_color = lambda vals: 'green' if list(vals)[0]=='1' else 'red'
        bar_colors = [bar_color(v) for v in plot_text.values()]
        bar_plot = ax.barh(np.arange(len(plot_data)), plot_data.values(), align='center', color=bar_colors)
    else:
        bar_plot = ax.barh(np.arange(len(plot_data)), plot_data.values(), align='center')
    ax.set_yticks(np.arange(len(plot_data)))
    ax.set_yticklabels(plot_data.keys())
    if y_label:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel('Transitions')
    if x_label:
        ax.set_xlabel(x_label)
    else:
        ax.set_xlabel('Frequency')
    if plot_title:
        ax.set_title(plot_title)

    # show text on bars
    for rect, text_set in zip(bar_plot, plot_text.values()):
        ax.text(0,
                rect.get_y() + rect.get_height()/2,
                ' // '.join(list(text_set)),
                ha='left',
                va='center',
                rotation=0)

def recc_dict():
    return defaultdict(recc_dict)

# init keys for all potential keys
def potential_actions(num_blocks):
    pos_actions = []
    neg_actions = []
    for bb in range(1, num_blocks+1):
        for bt in range(1, num_blocks+1):
            if bt == bb+1:
                pos_actions.append(str(bb)+','+str(bt))
            elif bt != bb:
                neg_actions.append(str(bb)+','+str(bt))
    return pos_actions, neg_actions

def plot_results(success_data, all_test_num_blocks, title, xlabel, ylabel, logger):
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
    axis.set_ylim(0, 1.1)
    axis.legend(title='Method')

    plt.savefig('%s/%s.png' % (logger.exp_path, title))
    print('Saving figures to %s.' % logger.exp_path)
