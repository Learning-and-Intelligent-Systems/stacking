import torch

from tamp.logic import subset
from learning.domains.abc_blocks.world import get_vectorized_state, N_OBJECTS, TABLE

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
            state = world.get_state()
            #if subset(goal, state):
            #    action_sequence.append((state, NONACTION))
            #    break
            #else:
            action = policy()
            world.transition(action)
            action_sequence.append((state, action))
            if len(world._stacked_blocks) == N_OBJECTS-1:
                state = world.get_state()
                #if subset(goal, state):
                #    action_sequence.append((state, NONACTION))
                #    break
                #else:
                action = policy()
                world.transition(action)
                action_sequence.append((state, action))
                break
        add_sequence_to_dataset(args, trans_dataset, heur_dataset, action_sequence, goal, logger)
    #logger.save_dataset(dataset, i)

def add_sequence_to_dataset(args, trans_dataset, heur_dataset, action_sequence, goal, logger):
    def helper(sequence, seq_goal):
        n = len(sequence)
        goal_object_features, goal_edge_features = get_vectorized_state(seq_goal)
        for i in range(n):
            state, action = sequence[i]
            object_features, edge_features = get_vectorized_state(state)
            heur_dataset.add_to_dataset(object_features, edge_features, goal_edge_features, n-i-1)
            if i < n-1: # training transition model doesn't require last action in sequence
                next_state, _ = sequence[i+1]
                next_object_features, next_edge_features = get_vectorized_state(next_state)
                if args.pred_type == 'delta_state':
                    edge_features_to_add = next_edge_features-edge_features
                elif args.pred_type == 'full_state':
                    edge_features_to_add = next_edge_features
                trans_dataset.add_to_dataset(object_features, edge_features, 
                                            action, edge_features_to_add)
                
    # if given goal was reached, add to dataset
    #if subset(goal, action_sequence[-1][0]):
    #helper(action_sequence, goal)
        
    # for all other reached states, make them goals (hindsight experience replay)
    for goal_i, (hindsight_goal, _) in enumerate(action_sequence):
        helper(action_sequence[:goal_i+1], hindsight_goal)
        
# for testing, only keep samples with successful actions/edge changes
def preprocess(args, dataset, type='successful_actions'):
    xs, ys = dataset[:]
    remove_list = []
    if type == 'successful_actions':
        for i, ((object_features, edge_features, action), next_edge_features) in enumerate(dataset):
            if (args.pred_type == 'full_state' and (edge_features == next_edge_features).all()) or \
                (args.pred_type == 'delta_state' and (next_edge_features.abs().sum() == 0)):
                remove_list.append(i)
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