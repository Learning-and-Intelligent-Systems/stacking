import argparse
import copy
from tamp.misc import get_train_and_fit_blocks, load_blocks
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import torch

from block_utils import Object, World, Environment
from learning.active.utils import ActiveExperimentLogger
from learning.domains.towers.active_utils import get_sequential_predictions
from learning.domains.towers.tower_data import ParallelDataLoader, TowerDataset
from learning.evaluate.planner import LatentEnsemblePlanner
from tower_planner import TowerPlanner


def get_validation_accuracy(logger, fname):
    """ Given a logger object, get the validation accuracy on a test set
    given at every timestep, separated by tower size.

    :param logger: Logger obejct describing the training set.
    :param fname: Path to the validation dataset.
    """
    tower_keys = ['2block', '3block', '4block', '5block']
    accs = {k: [] for k in tower_keys}
    
    with open(fname, 'rb') as handle:
        val_towers = pickle.load(handle)

    for tx in range(0, logger.args.max_acquisitions):
        print('Eval timestep, ', tx)
        ensemble = logger.get_ensemble(tx)
        preds = get_sequential_predictions(val_towers, ensemble, use_latents=logger.use_latents).mean(1).numpy()

        start = 0
        for k in tower_keys:
            end = start + val_towers[k]['towers'].shape[0]
            acc = ((preds[start:end]>0.5) == val_towers[k]['labels']).mean()
            print('Acc:', tx, k, acc)
            accs[k].append(acc)
            start = end
        
    with open(logger.get_figure_path('val_accuracies.pkl'), 'wb') as handle:
        pickle.dump(accs, handle)
    return accs


def plot_val_accuracy(logger, init_dataset_size=100, n_acquire=10, separate_axs=False):
    with open(logger.get_figure_path('val_accuracies.pkl'), 'rb') as handle:
        accs = pickle.load(handle)

    tower_keys = ['2block', '3block', '4block', '5block']    

    if separate_axs:
        plt.clf()
        fig, axes = plt.subplots(4, figsize=(10, 20))
        for ix, ax in enumerate(axes):
            k = tower_keys[ix]
            max_x = init_dataset_size + n_acquire*len(accs[k])
            xs = np.arange(init_dataset_size, max_x, n_acquire)

            ax.plot(xs, accs[k], label=k)
            ax.set_xlabel('Number of Towers')
            ax.set_ylabel('Val Accuracy')
            ax.legend()
    else:
        for k in tower_keys:
            max_x = init_dataset_size + n_acquire*len(accs[k])
            xs = np.arange(init_dataset_size, max_x, n_acquire)
            plt.plot(xs, accs[k], label=k)
        plt.xlabel('Number of Towers')
        plt.ylabel('Val Accuracy')
        plt.legend()

    plt.title('Validation Accuracy throughout active learning')
    plt.savefig(logger.get_figure_path('val_accuracy.png'))
    plt.clf()

def get_predictions_with_particles(particles, observation, ensemble):
    preds = []
    dataset = TowerDataset(tower_dict=observation,
                           augment=False)
    dataloader = ParallelDataLoader(dataset=dataset,
                                    batch_size=64,
                                    shuffle=False,
                                    n_dataloaders=1)
    

    latent_samples = torch.Tensor(particles)
    for set_of_batches in dataloader:
        towers, block_ids, _ = set_of_batches[0]
        sub_tower_preds = []
        
        for n_blocks in range(2, towers.shape[1]+1):
            if torch.cuda.is_available():
                tensor = tensor.cuda()
            with torch.no_grad():
                for ix in range(0, 1): # Only use first 10 particles as a sample from the particle distribution.
                    pred = ensemble.forward(towers=towers[:, :n_blocks, 4:],
                                            block_ids=block_ids[:, :n_blocks],
                                            N_samples=10,
                                            collapse_latents=True, 
                                            collapse_ensemble=True,
                                            pf_latent_ix=10,
                                            latent_samples=latent_samples[ix*10:(ix+1)*10,:]).squeeze()
                    sub_tower_preds.append(pred)

        sub_tower_preds = torch.stack(sub_tower_preds, dim=0)
        preds.append(sub_tower_preds.prod(dim=0))
    return torch.cat(preds, dim=0)

def get_pf_validation_accuracy(logger, fname):
    tower_keys = ['2block', '3block', '4block', '5block']
    accs = {k: [] for k in tower_keys}
    
    with open(fname, 'rb') as handle:
        val_towers = pickle.load(handle)

    for tx in range(0, logger.args.max_acquisitions):
        print('Eval timestep, ', tx)
        ensemble = logger.get_ensemble(tx)
        particles = logger.load_particles(tx)

        start = 0
        preds_list = []

        preds = get_predictions_with_particles(particles.particles, val_towers, ensemble)
        preds_list += preds.cpu().numpy().tolist()

        preds = np.array(preds_list)

        for k in tower_keys:
            end = start + val_towers[k]['towers'].shape[0]
            # for ix in range(0, val_towers[k]['towers'].shape[0]):
            #     td = { k: {
            #             'towers': val_towers[k]['towers'][ix:ix+1, :, :],
            #             'block_ids': val_towers[k]['block_ids'][ix:ix+1, :],
            #             'labels': val_towers[k]['labels'][ix:ix+1],
            #         }
            #     }
            # TODO: Right now we're averaging the particles before the prediction, we should be marginalizing the predictions.
            # latent = np.array(particles.particles).T@np.array(particles.weights)
            # latent = latent.reshape(1, 4)
            # latent = np.array(particles.particles)[0:10, 4]
            # print(latent.shape)
            
            acc = ((preds[start:end]>0.5) == val_towers[k]['labels']).mean()
            print('Acc:', tx, k, acc)
            accs[k].append(acc)
            start = end
        
    
    with open(logger.get_figure_path('val_accuracies.pkl'), 'wb') as handle:
        pickle.dump(accs, handle)
    return accs


def tallest_tower_regret_evaluation(logger, block_set, fname, args, save_imgs=False):
    def tower_height(tower):
        """
        :param tower: A vectorized version of the tower.
        """
        return np.sum(tower[:, 6])

    return evaluate_planner(logger, block_set, tower_height, fname, args, save_imgs, img_prefix='height')

def cumulative_overhang_regret_evaluation(logger, block_set, fname, args, save_imgs=False):
    def horizontal_overhang(tower):
        total_overhang = 0
        for tx in range(1, tower.shape[0]):
            bx = tx - 1
            overhang = (tower[tx, 7] + tower[tx, 4]/2.) - (tower[bx, 7] + tower[bx, 4]/2.)
            total_overhang += overhang
        return total_overhang
    
    return evaluate_planner(logger, block_set, horizontal_overhang, fname, args, save_imgs, img_prefix='cumulative_overhang')
    
def longest_overhang_regret_evaluation(logger, block_set, fname, args, save_imgs=False):
    def horizontal_overhang(tower):
        return (tower[-1, 7] + tower[-1, 4]/2.) - (tower[0, 7] + tower[0, 4]/2.)
    
    return evaluate_planner(logger, block_set, horizontal_overhang, fname, args, save_imgs, img_prefix='overhang')
    
def min_contact_regret_evaluation(logger, block_set, fname, args, save_imgs=False):
    def contact_area(tower):
        """
        :param tower: A vectorized version of the tower.
        """
        lefts, rights = tower[:, 7] - tower[:, 4]/2., tower[:, 7] + tower[:, 4]/2.
        bottoms, tops = tower[:, 8] - tower[:, 5]/2., tower[:, 8] + tower[:, 5]/2.

        area = 0.
        for tx in range(1, tower.shape[0]):
            bx = tx - 1
            l, r = max(lefts[bx], lefts[tx]), min(rights[bx], rights[tx])
            b, t = max(bottoms[bx], bottoms[tx]), min(tops[bx], tops[tx])
            
            top_area = tower[tx, 4]*tower[tx, 5]
            contact = (r-l)*(t-b)
            area += top_area - contact

        return area

    return evaluate_planner(logger, block_set, contact_area, fname, args, save_imgs, img_prefix='contact')

def evaluate_planner(logger, blocks, reward_fn, fname, args, save_imgs=False, img_prefix=''):
    tower_keys, tower_sizes  = ['5block'], [5]
    tp = TowerPlanner(stability_mode='contains')
    ep = LatentEnsemblePlanner(logger)

    # Store regret for towers of each size.
    regrets = {k: [] for k in tower_keys}
    rewards = {k: [] for k in tower_keys}

    if args.max_acquisitions is not None: 
        eval_range = range(0, args.max_acquisitions, 2)
    elif args.acquisition_step is not None: 
        eval_range = [args.acquisition_step]
    
    for tx in eval_range:
        print('Acquisition step:', tx)

        ensemble = logger.get_ensemble(tx)
        particles = logger.load_particles(tx)
        if particles is not None:
            latent_samples = torch.Tensor(particles.particles)[0:10, :]
            pf_latent_ix = 10
        else:
            latent_samples = None
            pf_latent_ix = -1
        if torch.cuda.is_available():
            ensemble = ensemble.cuda()
            
        for k, size in zip(tower_keys, tower_sizes):
            print('Tower size', k)
            num_failures, num_pw_failures = 0, 0
            curr_regrets = []
            curr_rewards = []
            for t in range(0, args.n_towers):
                print('Tower number', t)
                
                if ('fit' in logger.args and logger.args.fit) or (logger.load_particles(0) is not None):
                    # Must include the fitting block.
                    plan_blocks = [block_set[-1]] + list(np.random.choice(block_set[:-1], size-1, replace=False))
                    plan_blocks = copy.deepcopy(plan_blocks)
                    print('Planning with blocks: ', [b.name for b in plan_blocks])
                else:
                    plan_blocks = np.random.choice(blocks, size, replace=False)	
                    plan_blocks = copy.deepcopy(plan_blocks)	
                
                tower, reward, max_reward, tower_block_ids, rotated_tower = ep.plan(plan_blocks, 
                                                                     ensemble, 
                                                                     reward_fn,
                                                                     args,
                                                                     num_blocks=size,
                                                                     n_tower=t,
                                                                     pf_latent_ix=pf_latent_ix,
                                                                     latent_samples=latent_samples)
                                                                 
                # perturb tower if evaluating with noisy model
                block_tower = []
                for vec_block, block_id in zip(rotated_tower, tower_block_ids):
                    vec_block[7:9] += np.random.randn(2)*args.exec_xy_noise
                    block = Object.from_vector(vec_block)
                    block.name = 'obj_%d' % block_id
                    block_tower.append(block)     
    
                # build found tower
                if not tp.tower_is_constructable(block_tower):
                    reward = 0
                    num_failures += 1

                    if False and reward != 0:
                        print(reward, max_reward)
                        w = World(block_tower)
                        env = Environment([w], vis_sim=True, vis_frames=True)
                        input()
                        for tx in range(240):
                            env.step(vis_frames=True)
                            time.sleep(1/240.)
                        env.disconnect()

                # Compare heights and calculate regret.
                regret = (max_reward - reward)/max_reward
                curr_regrets.append(regret)
                curr_rewards.append(reward)
                print('Reward: %.2f\tMax: %.2f\tRegret: %.2f' % (reward, max_reward, regret))

            regrets[k].append(curr_regrets)
            rewards[k].append(curr_rewards)

        if args.max_acquisitions is not None:
            with open(logger.get_figure_path(fname+'_regrets.pkl'), 'wb') as handle:
                pickle.dump(regrets, handle)
                
            with open(logger.get_figure_path(fname+'_rewards.pkl'), 'wb') as handle:
                pickle.dump(rewards, handle)
                
            plot_regret(logger, args.problem + '_regrets.pkl')
        else:
            with open(logger.get_figure_path(fname+'_%d_regrets.pkl' % args.acquisition_step), 'wb') as handle:
                pickle.dump(regrets, handle)
                
            with open(logger.get_figure_path(fname+'_%d_rewards.pkl' % args.acquisition_step), 'wb') as handle:
                pickle.dump(rewards, handle)
            
    # if just ran for one acquisition step, output final regret and reward
    if args.acquisition_step is not None:
        final_median_regret = np.median(regrets[k][0])
        final_upper75_regret = np.quantile(regrets[k][0], 0.75)
        final_lower25_regret = np.quantile(regrets[k][0], 0.25)
        
        final_median_reward = np.median(rewards[k][0])
        final_upper75_reward = np.quantile(rewards[k][0], 0.75)
        final_lower25_reward = np.quantile(rewards[k][0], 0.25)
        
        final_average_regret = np.average(regrets[k][0])
        final_std_regret = np.std(regrets[k][0])
        
        final_average_reward = np.average(rewards[k][0])
        final_std_reward = np.std(rewards[k][0])
        
        print('Final Median Regret: (%f) %f (%f)' % (final_lower25_regret, final_median_regret, final_upper75_regret))
        print('Final Median Reward: (%f) %f (%f)' % (final_lower25_reward, final_median_reward, final_upper75_reward))
        
        print('Final Average Regret: %f +/- %f' % (final_average_regret, final_std_regret))
        print('Final Average Reward: %f +/- %f' % (final_average_reward, final_std_reward))


def plot_regret(logger, fname):
    with open(logger.get_figure_path(fname), 'rb') as handle:
        regrets = pickle.load(handle)

    tower_keys = ['5block'] # ['2block', '3block', '4block', '5block']
    upper975 = {k: [] for k in tower_keys}
    upper75 = {k: [] for k in tower_keys}
    median = {k: [] for k in tower_keys}
    lower25 = {k: [] for k in tower_keys}
    lower025 = {k: [] for k in tower_keys}
    for k in tower_keys:
        rs = regrets[k]
        #print(rs)
        for tx in range(len(rs)):
            median[k].append(np.median(rs[tx]))
            lower025[k].append(np.quantile(rs[tx], 0.05))
            lower25[k].append(np.quantile(rs[tx], 0.25))
            upper75[k].append(np.quantile(rs[tx], 0.75))
            upper975[k].append(np.quantile(rs[tx], 0.95))
    fig, axes = plt.subplots(4, sharex=True, figsize=(10,20))
    init, n_acquire = logger.get_acquisition_params()
    for kx, k in enumerate(tower_keys):
        xs = np.arange(init, init+2*len(median[k]), 2*n_acquire)
        axes[kx].plot(xs, median[k], label=k)
        axes[kx].fill_between(xs, lower25[k], upper75[k], alpha=0.2)
        axes[kx].set_ylim(0.0, 1.1)
        axes[kx].set_ylabel('Regret (Normalized)')
        axes[kx].set_xlabel('Number of training towers')
        axes[kx].legend()
    plt_fname = fname[:-4]+'.png'
    plt.savefig(logger.get_figure_path(plt_fname))


# Common datasets to use:
# Fitting: 'learning/data/may_blocks/towers/combined_traineval0.pkl', 'learning/data/may_blocks/towers/combined_traineval1.pkl'
# Training: 'learning/data/may_blocks/towers/10block_set_(x1000)_nblocks_a_1_dict.pkl'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-path', type=str, required=True)
    parser.add_argument('--eval-type', type=str, choices=['val', 'task'], required=True)
    # Validation evaluation arguments.
    parser.add_argument('--val-dataset-fname', type=str)
    # Task performance arguments.
    parser.add_argument('--problem', type=str, choices=['min-contact', 'cumulative-overhang', 'tallest'])
    parser.add_argument('--max-acquisitions',
                        type=int, 
                        help='evaluate from 0 to this acquisition step (use either this or --acquisition-step)')
    parser.add_argument('--acquisition-step',
                        type=int,
                        help='acquisition step to evaluate (use either this or --max-acquisition)')
    parser.add_argument('--n-towers',
                        default = 50,
                        type=int,
                        help = 'number of tall towers to find for each acquisition step')
    parser.add_argument('--exec-xy-noise',
                        type=float,
                        help='noise to add to xy position of blocks if exec-mode==noisy-model')
    args = parser.parse_args()
    
    logger = ActiveExperimentLogger(args.exp_path, use_latents=True)

    if args.eval_type == 'val':
        init_dataset_size, n_acquire = logger.get_acquisition_params()
        if logger.load_particles(tx=0) is not None:
            accs = get_pf_validation_accuracy(logger,
                                              fname=args.val_dataset_fname)
        else:
            accs = get_validation_accuracy(logger=logger,
                                           fname=args.val_dataset_fname)                     
        plot_val_accuracy(logger, init_dataset_size=init_dataset_size, n_acquire=n_acquire)

    elif args.eval_type == 'task':
        # Load the block set for evaluation.
        if ('fit' in logger.args and logger.args.fit) or (logger.load_particles(0) is not None):
            block_set = get_train_and_fit_blocks(pretrained_ensemble_path=logger.args.pretrained_ensemble_exp_path,
                                                 use_latents=True,
                                                 fit_blocks_fname=logger.args.block_set_fname,
                                                 fit_block_ixs=logger.args.eval_block_ixs)
        else:
            block_set = load_blocks(train_blocks_fname=logger.args.block_set_fname)

        reward_lookup = {
            'min-contact': min_contact_regret_evaluation,
            'tallest': tallest_tower_regret_evaluation,
            'cumulative-overhang': cumulative_overhang_regret_evaluation
        }

        # TODO: If fitting, make sure to always include the new block in the block set.
        reward_lookup[args.problem](logger=logger,
                                    block_set=block_set,
                                    fname=args.problem,
                                    args=args)            

        


