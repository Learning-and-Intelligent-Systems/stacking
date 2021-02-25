import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
matplotlib.use("TkAgg")

from tower_planner import TowerPlanner
from learning.active.utils import ActiveExperimentLogger

n_towers_to_visualize = 5
def visualize_towers(tower_data, args):
    tp = TowerPlanner()
    for tower_file, towers in tower_data.items():
        for k in args.tower_sizes:
            k_tower_data = towers['%dblock' % k]
            for tower, reward, max_reward in k_tower_data[:n_towers_to_visualize]:
                n_blocks = len(tower)
                constructable = tp.tower_is_constructable(tower)
                print(constructable, reward, max_reward)
                fig, axes = plt.subplots(n_blocks-1)
                for bbi in range(1,n_blocks):
                    # for each subtower (starting from the tower above the bottom block
                    # and ending in the top block)
                    # plot the combined com in the contact patch of the support block
                    combined_com = np.zeros(3)
                    for sub_block in tower[bbi:]:
                        combined_com += sub_block.mass*np.array(sub_block.com)
                    axes[bbi-1].plot(*combined_com[:2], 'r*')
                    support_block = tower[bbi-1]
                    rect = patches.Rectangle((0,0), support_block.dimensions.y, support_block.dimensions.x, linewidth=1, edgecolor='k', facecolor='none')
                    axes[bbi-1].add_patch(rect)
                    axes[bbi-1].set_title('Block %d Support' % (bbi))
                    axes[bbi-1].set_aspect('equal', adjustable='box')
                fig.suptitle(tower_file)
                fig.tight_layout()
                plt.show()
    
def plot_block_placements(args):
    logger = ActiveExperimentLogger(args.exp_path)
    for task in ['tallest', 'overhang', 'min-contact']:
        for planning_model in ['learned', 'noisy-model', 'simple-model']:
            try:
                towers_data = logger.get_evaluation_towers(task, planning_model, args.acquisition_step)
            except:
                continue
            visualize_towers(towers_data, args)
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-path', 
                        type=str,
                        help='evaluate from 0 to this acquisition step (use either this or --acquisition-step)')
    parser.add_argument('--acquisition-step',
                        type=int,
                        help='acquisition step to evaluate (use either this or --max-acquisition)')
    parser.add_argument('--debug',
                        action='store_true',
                        help='set to run in debug mode')
    parser.add_argument('--tower-sizes',
                        default=[5],
                        type=int,
                        nargs='+',
                        help='number of blocks in goal tower (can do multiple)')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()
        
    plot_block_placements(args)