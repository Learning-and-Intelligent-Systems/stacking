import torch
import numpy as np

from tamp.logic import subset

# TODO: generate validation dataset too!!
# want to show that this will be improved by an actively collected dataset
def generate_dataset(args, world, logger, trans_dataset, heur_dataset, policy):
    goal = None
    for seq_i in range(args.max_seq_attempts):
        new_state = world.get_init_state()
        action_sequence = []
        valid_actions = True
        action_i = 0
        while valid_actions and action_i < args.max_action_attempts:
            state = new_state
            vec_action = policy(state)
            new_state = world.transition(state, vec_action)
            action_sequence.append((state.as_vec(), vec_action))
            action_i += 1
            # no valid actions left
            if world.expert_policy(new_state) is None:
                valid_actions = False
                action = np.zeros(2)
                action_sequence.append((new_state.as_vec(), vec_action))
        add_sequence_to_dataset(args, trans_dataset, heur_dataset, action_sequence, goal, logger)

def add_sequence_to_dataset(args, trans_dataset, heur_dataset, action_sequence, goal, logger):
    def helper(sequence, vec_seq_goal):
        n = len(sequence)
        object_features, goal_edge_features = vec_seq_goal
        for i in range(n):
            vec_state, vec_action = sequence[i]
            object_features, edge_features = vec_state
            heur_dataset.add_to_dataset(object_features, edge_features, goal_edge_features, n-i-1)
            if i < n-1: # training transition model doesn't require last action in sequence
                vec_next_state, _ = sequence[i+1]
                object_features, next_edge_features = vec_next_state
                if args.pred_type == 'delta_state':
                    edge_features_to_add = next_edge_features-edge_features
                elif args.pred_type == 'full_state':
                    edge_features_to_add = next_edge_features
                trans_dataset.add_to_dataset(object_features, edge_features, vec_action, edge_features_to_add)

    # for all other reached states, make them goals (hindsight experience replay)
    #for goal_i, (hindsight_goal, _) in enumerate(action_sequence):
    #    helper(action_sequence[:goal_i+1], hindsight_goal)
    # TODO: for now just add each transition to dataset once. when learning heuristics
    # will add hindsight goals
    final_goal, final_action = action_sequence[-1]
    helper(action_sequence, final_goal)


# for testing
def preprocess(args, dataset, type='successful_actions'):
    xs, ys = dataset[:]
    remove_list = []
    # only keep samples with successful actions/edge changes
    if type == 'successful_actions':
        for i, ((object_features, edge_features, action), next_edge_features) in enumerate(dataset):
            if (args.pred_type == 'full_state' and (edge_features == next_edge_features).all()) or \
                (args.pred_type == 'delta_state' and (next_edge_features.abs().sum() == 0)):
                remove_list.append(i)
    # all actions have same frequency in the dataset
    if type == 'balanced_actions':
        distinct_actions = []
        actions_counter = {}
        for i, ((object_features, edge_features, action), next_edge_features) in enumerate(dataset):
            a = tuple(action.numpy())
            if a not in distinct_actions:
                distinct_actions.append(a)
                actions_counter[a] = [i]
            else:
                actions_counter[a] += [i]
        min_distinct_actions = min([len(counter) for counter in actions_counter.values()])
        for a in distinct_actions:
            remove_list += actions_counter[a][min_distinct_actions:]

    dataset.remove_elements(remove_list)
