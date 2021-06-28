import torch

from tamp.logic import subset

# TODO: generate validation dataset too!!
def generate_dataset(args, world, logger, trans_dataset, heur_dataset, policy):
    #for gi, goal in enumerate(world.parse_goals_csv(args.goals_file_path)):
        #print('Generating data for goal %i' % gi)
    #goal = world.parse_goals_csv(args.goals_file_path)[0]
    goal = None
    for i in range(args.max_seq_attempts):
        world.reset()
        action_sequence = []
        for j in range(args.max_action_attempts):
            vec_state = world.get_vectorized_state()
            #if subset(goal, state):
            #    action_sequence.append((state, NONACTION))
            #    break
            #else:
            action = policy()
            vec_action = world.get_vectorized_action(action)
            world.transition(action)
            action_sequence.append((vec_state, vec_action))
            if len(world._stacked_blocks) == world.num_objects-1:
                vec_state = world.get_vectorized_state()
                #if subset(goal, state):
                #    action_sequence.append((state, NONACTION))
                #    break
                #else:
                action = policy()
                vec_action = world.get_vectorized_action(action)
                world.transition(action)
                action_sequence.append((vec_state, vec_action))
                break
        add_sequence_to_dataset(args, trans_dataset, heur_dataset, action_sequence, goal, logger)
    #logger.save_dataset(dataset, i)

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
                
    # if given goal was reached, add to dataset
    #if subset(goal, action_sequence[-1][0]):
    #helper(action_sequence, goal)
        
    # for all other reached states, make them goals (hindsight experience replay)
    for goal_i, (hindsight_goal, _) in enumerate(action_sequence):
        helper(action_sequence[:goal_i+1], hindsight_goal)
        
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