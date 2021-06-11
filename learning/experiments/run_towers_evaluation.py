import os
import pdb
import pickle
import argparse
from agents.panda_agent import PandaAgent
from tamp.misc import load_blocks
import time

def main(args):

    # Load the tower data
    with open(args.towers_file, "rb") as f:
        towers_data = pickle.load(f)
        n_towers = sum([len(towers_data[k]) for k in towers_data])
        print(f"Evaluating on {n_towers} towers ...")
    
    # Check if a labeled file already exists
    labeled_file = args.towers_file[:-4] + "_labeled.pkl"
    if os.path.exists(labeled_file):
        with open(labeled_file, "rb") as f:
            labeled_data = pickle.load(f)
            n_labeled = sum([len(labeled_data[k]) for k in labeled_data])
            print(f"Found existing labeled file with {n_labeled} towers")
    else:
        print(f"Creating new file: {labeled_file}")
        labeled_data = {}
        for k in towers_data:
            labeled_data[k] = []
        n_labeled = 0

    # Instantiate the tower execution agent
    blocks = load_blocks(fname=args.blocks_file,
                         num_blocks=args.num_blocks)
    agent = PandaAgent(blocks,
                      use_platform=False, 
                      use_planning_server=args.use_planning_server,
                      use_vision=args.use_vision,
                      real=args.real)

    # Loop through all the towers in the list
    for k in towers_data:
        for tx, data in enumerate(towers_data[k]):
            # Check if tower was already labeled
            if tx < n_labeled:
                print(f"Already labeled tower {tx}")
                continue
            
            # Instruct the agent to build the tower
            tower, max_reward, reward = data
            n_blocks = len(tower)
            print(f"Starting tower {tx}")
            success, stable, n_stacked = agent.simulate_tower(tower,
                                                              real=args.real,
                                                              base_xy=(0.5, -0.3),
                                                              vis=True,
                                                              T=2500,
                                                              ignore_resets=True)
            # If successful, add the stability information to the list
            if success:
                print(f"Finished {k} tower {tx} with stable: {stable}, num successful: {n_stacked}/{n_blocks}")
                labeled_data[k].append((tower, max_reward, reward, stable, n_stacked))
                with open(labeled_file, "wb") as f:
                    pickle.dump(labeled_data, f)
                input("Reset the world and press Enter to continue.")
            else:
                input("Failed! Reset the world and press Enter to try again")

            # Reinitialize the planning state of the world
            agent.reset_world()

    print("\nEvaluations complete!\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--use-planning-server', action='store_true', help='Use planning server')
    parser.add_argument('--use-vision', action='store_true', help='Get block poses from AR tags')
    parser.add_argument('--blocks-file', type=str, default='learning/domains/towers/final_block_set_10.pkl')
    parser.add_argument('--towers-file', type=str, default='learning/experiments/towers_40.pkl')
    parser.add_argument('--num-blocks', type=int, default=10)
    parser.add_argument('--real', action='store_true', help='run on real robot')
    args = parser.parse_args()
    if args.debug: pdb.set_trace()

    main(args)
