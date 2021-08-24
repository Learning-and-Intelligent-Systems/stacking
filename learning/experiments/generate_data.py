import torch
import numpy as np
import argparse
from argparse import Namespace
from torch.nn import functional as F
from torch.utils.data import DataLoader

from tamp.logic import subset
from learning.domains.abc_blocks.world import ABCBlocksWorldGT, generate_random_goal
from learning.domains.abc_blocks.abc_blocks_data import ABCBlocksTransDataset, ABCBlocksHeurDataset
from learning.active.utils import GoalConditionedExperimentLogger
from learning.models.goal_conditioned import TransitionGNN
from planning import plan
from learning.active.train import train

# TODO: this doesn't make len(dataset) == args.max_transitions exactly
# because sequences are added in chunks that might push it past the limit
# but will be close
plan_args = argparse.Namespace(num_branches=10,
                        timeout=100,
                        c=.01,
                        max_ro=10,
                        exp_name='data_gen')
model_args = Namespace(pred_type='class',
                        batch_size=16,
                        n_epochs=300,
                        n_hidden = 16)

def random_goals_dataset(args, world, trans_dataset, heur_dataset, dataset_logger):
    # save initial (empty) dataset
    logger.save_trans_dataset(trans_dataset, i=0)

    # initialize and save model
    trans_model = TransitionGNN(n_of_in=1,
                                n_ef_in=1,
                                n_af_in=2,
                                n_hidden=16,
                                pred_type='class')
    model_args.dataset_exp_path = logger.exp_path
    model_args.exp_name = 'model-%s' % args.exp_name
    model_args.num_blocks = world.num_blocks
    model_logger = GoalConditionedExperimentLogger.setup_experiment_directory(model_args, 'models')
    model_logger.save_trans_model(trans_model, i=0)

    # initialize planner
    plan_args.num_blocks = world.num_blocks
    if args.mode == 'random-goals-opt':
        plan_args.model_type = 'opt'
    elif args.mode == 'random-goals-learned':
        plan_args.model_type = 'learned'
        plan_args.model_exp_path = model_logger.exp_path
    plan_args.value_fn = 'rollout'

    i = 0
    while len(trans_dataset) < args.max_transitions:
        print('Run %i |dataset| = %i' % (i, len(trans_dataset)))
        # generate plan to reach random goal
        goal = generate_random_goal(world)
        found_plan, plan_exp_path, rank_accuracy = plan.run(goal, plan_args, model_i=i)
        if found_plan:
            trajectory = world.execute_plan(found_plan)
        else:
            trajectory = random_action_sequence(world)

        # add to dataset and save
        print('Adding trajectory to dataset.')
        add_sequence_to_dataset(args, trans_dataset, heur_dataset, trajectory, goal, world)

        # initialize and train new model
        trans_model = TransitionGNN(n_of_in=1,
                                    n_ef_in=1,
                                    n_af_in=2,
                                    n_hidden=16,
                                    pred_type='class')
        trans_dataset.set_pred_type('class')
        if len(trans_dataset) > 0:
            print('Training model.')
            trans_dataloader = DataLoader(trans_dataset, batch_size=model_args.batch_size, shuffle=True)
            train(trans_dataloader, None, trans_model, n_epochs=model_args.n_epochs, loss_fn=F.binary_cross_entropy)
        # save new model and dataset
        i += 1
        logger.save_trans_dataset(trans_dataset, i=i)
        model_logger.save_trans_model(trans_model, i=i)
        print('Saved model to %s' % model_logger.exp_path)

def random_action_dataset(args, world, trans_dataset, heur_dataset):
    goal = None
    while len(trans_dataset) < args.max_transitions:
        action_sequence = random_action_sequence(world)
        add_sequence_to_dataset(args, trans_dataset, heur_dataset, action_sequence, goal, world)

def random_action_sequence(world):
    new_state = world.get_init_state()
    valid_actions = True
    action_sequence = []
    while valid_actions:
        state = new_state
        vec_action = world.random_policy(state)
        new_state = world.transition(state, vec_action)
        action_sequence.append((state, vec_action))
        valid_actions = world.expert_policy(new_state) is not None
    vec_action = np.zeros(2)
    action_sequence.append((new_state, vec_action))
    return action_sequence


def add_sequence_to_dataset(args, trans_dataset, heur_dataset, action_sequence, goal, world):
    def helper(sequence, seq_goal, add_to_trans):
        n = len(sequence)
        object_features, goal_edge_features = seq_goal.as_vec()
        for i in range(n):
            state, vec_action = sequence[i]
            object_features, edge_features = state.as_vec()
            heur_dataset.add_to_dataset(object_features, edge_features, goal_edge_features, n-i-1)
            # add_to_trans: only add to transition model once per sequence (not hindsight subsequences)
            # i < n-1: don't add last action to trans dataset as it doesn't do anything
            if add_to_trans and i < n-1:
                next_state, _ = sequence[i+1]
                object_features, next_edge_features = next_state.as_vec()
                delta_edge_features = next_edge_features-edge_features
                optimistic_accuracy = 1 if world.transition(state, vec_action, optimistic=True).is_equal(next_state) \
                                        else 0
                trans_dataset.add_to_dataset(object_features, edge_features, vec_action, next_edge_features, delta_edge_features, optimistic_accuracy)

    # make each reached state a goal (hindsight experience replay)
    for goal_i, (hindsight_goal, _) in enumerate(action_sequence):
        helper(action_sequence[:goal_i+1], hindsight_goal, goal_i == len(action_sequence)-1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',
                        action='store_true',
                        help='set to run in debug mode')
    parser.add_argument('--domain',
                        type=str,
                        choices=['abc_blocks', 'com_blocks'],
                        default='abc_blocks',
                        help='domain to generate data from')
    parser.add_argument('--max-transitions',
                        type=int,
                        default=300,
                        help='max number of transitions to save to transition dataset')
    parser.add_argument('--exp-name',
                        type=str,
                        required=True,
                        help='path to save dataset to')
    parser.add_argument('--num-blocks',
                        type=int,
                        default=3)
    parser.add_argument('--mode',
                        type=str,
                        choices=['random-actions', 'random-goals-opt', 'random-goals-learned'],
                        required=True,
                        help='method of data collection')
    parser.add_argument('--N',
                        type=int,
                        default=1,
                        help='number of models to train')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    for _ in range(args.N):
        #try:
        if args.domain == 'abc_blocks':
            world = ABCBlocksWorldGT(args.num_blocks)
        else:
            NotImplementedError('Only ABC Blocks world works')

        trans_dataset = ABCBlocksTransDataset()
        heur_dataset = ABCBlocksHeurDataset()
        logger = GoalConditionedExperimentLogger.setup_experiment_directory(args, 'datasets')
        if args.mode == 'random-actions':
            random_action_dataset(args, world, trans_dataset, heur_dataset)
        elif args.mode in ['random-goals-opt', 'random-goals-learned']:
            random_goals_dataset(args, world, trans_dataset, heur_dataset, logger)
        logger.save_trans_dataset(trans_dataset)
        #logger.save_heur_dataset(heur_dataset)
        print('Datasets (N=%i) and tower info saved to %s.' % (len(trans_dataset), logger.exp_path))

        #except:
        #    import pdb; pdb.post_mortem()
