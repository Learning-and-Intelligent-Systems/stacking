import torch

from tamp.logic import subset
from learning.domains.abc_blocks.world import get_vectorized_state, NONACTION

# TODO: generate validation dataset too!!
def generate_dataset(args, world, logger, trans_dataset, heur_dataset, policy):
    #for gi, goal in enumerate(world.parse_goals_csv(args.goals_file_path)):
        #print('Generating data for goal %i' % gi)
    goal = world.parse_goals_csv(args.goals_file_path)[0]
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
        add_sequence_to_dataset(trans_dataset, heur_dataset, action_sequence, goal, logger)
    #logger.save_dataset(dataset, i)

def add_sequence_to_dataset(trans_dataset, heur_dataset, action_sequence, goal, logger):
    def helper(sequence, seq_goal):
        n = len(sequence)
        vec_goal = get_vectorized_state(seq_goal)
        for i in range(n):
            state, action = sequence[i]
            vec_state = get_vectorized_state(state)
            heur_dataset.add_to_dataset(vec_state, vec_goal, n-i-1)
            if i < n-1: # training transition model doesn't require last action in sequence
                next_state, _ = sequence[i+1]
                vec_next_state = get_vectorized_state(next_state)
                # NOTE: this results in -1 when edges should be deleted
                # so can't use for binary cross entropy
                delta_state = vec_next_state-vec_state
                #if delta_state.sum() != 0:
                #    print(delta_state)
                trans_dataset.add_to_dataset(vec_state, action, delta_state)
                #trans_dataset.add_to_dataset(vec_state, action, vec_next_state)
            
                
    # if given goal was reached, add to dataset
    #if subset(goal, action_sequence[-1][0]):
    #helper(action_sequence, goal)
        
    # for all other reached states, make them goals (hindsight experience replay)
    for goal_i, (hindsight_goal, _) in enumerate(action_sequence):
        helper(action_sequence[:goal_i+1], hindsight_goal)