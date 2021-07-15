import argparse

from planning import mcts
from learning.domains.abc_blocks.world import ABCBlocksWorldGT, ABCBlocksWorldLearned
from learning.active.utils import GoalConditionedExperimentLogger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-blocks',
                        type=int,
                        default=3,
                        help='only used in model_type == true')
    parser.add_argument('--num-branches',
                        type=int,
                        default=10,
                        help='number of actions to try from each node')
    parser.add_argument('--timeout',
                        type=int,
                        default=100,
                        help='Iterations to run MCTS')
    parser.add_argument('--c',
                        type=int,
                        default=.01,
                        help='UCB parameter to balance exploration and exploitation')
    parser.add_argument('--max-ro',
                        type=int,
                        default=10,
                        help='Maximum number of random rollout steps')
    parser.add_argument('--debug',
                        action='store_true',
                        help='set to run in debug mode')
    parser.add_argument('--model-type',
                        type=str,
                        choices=['learned', 'true'],
                        default='true',
                        help='plan with learned or ground truth model')
    parser.add_argument('--model-exp-path',
                        type=str,
                        help='Path to torch model to use during planning (if args.model_type == learned)')
    parser.add_argument('--exp-name',
                        type=str,
                        required=True,
                        help='experiment name for saving planning results')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    print('Planning with %s model.' % args.model_type)
    if args.model_type == 'learned':
        model_logger = GoalConditionedExperimentLogger(args.model_exp_path)
        model_args = model_logger.load_args()
        model = model_logger.load_model()
        world = ABCBlocksWorldLearned(model_args.num_blocks, model)
        print('Using model %s.' % model_logger.exp_path)
    elif args.model_type == 'true':
        world = ABCBlocksWorldGT(args.num_blocks)
        print('Planning with %i blocks.' % args.num_blocks)

    # for testing
    from tamp.predicates import On
    goal = [On(world._blocks[1], world._blocks[2]), On(world._blocks[2], world._blocks[3])]

    tree = mcts.run(world, goal, args)
    mcts.plan_from_tree(world, goal, tree)

    logger = GoalConditionedExperimentLogger.setup_experiment_directory(args, 'planning')
    logger.save_planning_data(tree, goal)
    print('Saved planning tree and goal to %s.' % logger.exp_path)
