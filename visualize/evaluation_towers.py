import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
matplotlib.use("TkAgg")

from tower_planner import TowerPlanner
from learning.active.utils import ActiveExperimentLogger
from block_utils import group_blocks, World, Environment

def plot_tower_stability(tower, title=''):
    n_blocks = len(tower)
    fig, axes = plt.subplots(n_blocks-1)
    top_group = tower[-1]

    for bbi, bottom_block in enumerate(reversed(tower[:-1])):
        # plot top group in bottom block
        top_com = np.add(top_group.pose.pos, top_group.com)
        axes[bbi].plot(*top_com[:2], 'r*')
        px, py = bottom_block.pose.pos.x, bottom_block.pose.pos.y
        h, w = bottom_block.dimensions.x, bottom_block.dimensions.y
        rect = patches.Rectangle((px-h/2,py-w/2), h, w, linewidth=1, edgecolor='k', facecolor='none')
        axes[bbi].add_patch(rect)
        #axes[bbi].set_title('Block %d from the Top Support' % (bbi+1))
        axes[bbi].set_aspect('equal', adjustable='box')
        
        # add the block to the group
        top_group = group_blocks(bottom_block, top_group)
    fig.suptitle('Blocks from the Top Down')
    fig.tight_layout()
    plt.show()
    
n_towers_to_visualize = 10
def visualize_towers(tower_data, args):
    tp = TowerPlanner(stability_mode='contains')
    for tower_file, towers in tower_data.items():
        for k in args.tower_sizes:
            k_tower_data = towers['%dblock' % k]
            for tower, reward, max_reward in k_tower_data[:n_towers_to_visualize]:
                #print(tower[0].pose, tower[0].rotation, tower[0].dimensions, tower[0].com, tower[0].name)
                constructable = tp.tower_is_constructable(tower)
                print(constructable, reward, max_reward)
                plot_tower_stability(tower, tower_file)
                
                # visualize in pybullet
                w = World(tower)
                env = Environment([w], vis_sim=True, vis_frames=True)
                input()
                for tx in range(240):
                    env.step(vis_frames=True)
                    time.sleep(1/240.)
                env.disconnect()
    
def plot_block_placements(args):
    logger = ActiveExperimentLogger(args.exp_path)
    for task in args.problems:
        for planning_model in args.planning_models:
            try:
                towers_data = logger.get_evaluation_towers(task, planning_model, args.acquisition_step)
            except:
                continue
            print(task, planning_model)
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
    parser.add_argument('--problems',
                        nargs='+',
                        type=str,
                        default=['min-contact']) # tallest overhang
    parser.add_argument('--planning-models',
                        nargs='+',
                        type=str,
                        default=['learned', 'simple-model']) # noisy-model
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()
        
    plot_block_placements(args)