import argparse
import numpy as np
from learning.active.utils import ActiveExperimentLogger

def cumulative_overhang(tower):
    total_overhang = 0
    for tx in range(1, tower.shape[0]):
        bx = tx - 1
        overhang = (tower[tx, 8] + tower[tx, 5]/2.) - (tower[bx, 8] + tower[bx, 5]/2.)
        total_overhang += overhang
    return total_overhang


def calc_reward_stable(results, args):
    rewards = []
    for entry in results['5block']:
        tower, reward, max_reward, stable, n_success = entry
        if not stable:
            reward = 0
        if stable:
            rewards.append(reward)

    return '%.2f' % np.mean(rewards)

def calc_regret_stable(results, args):
    regrets = []
    for entry in results['5block']:
        tower, reward, max_reward, stable, n_success = entry
        if not stable:
            reward = 0
        
        if stable:
            regret = (max_reward - reward)/max_reward
            regrets.append(regret)

    return '%.2f' % np.mean(regrets)

def calc_reward_partial(results, args):
    rewards = []
    for entry in results['5block']:
        tower, reward, max_reward, stable, n_success = entry
        tower_vec = np.array([b.vectorize() for b in tower[:n_success]])
        if args.problem == 'cumulative-overhang':
            reward = cumulative_overhang(tower_vec)
        else:
            raise NotImplementedError()
        rewards.append(reward)

    return '%.2f' % np.mean(rewards)

def calc_regret_partial(results, args):
    regrets = []
    for entry in results['5block']:
        tower, reward, max_reward, stable, n_success = entry
        tower_vec = np.array([b.vectorize() for b in tower[:n_success]])
        if args.problem == 'cumulative-overhang':
            reward = cumulative_overhang(tower_vec)
        else:
            raise NotImplementedError()
        regret = (max_reward - reward)/max_reward
        regrets.append(regret)

    return '%.2f' % np.mean(regrets)

def calc_regret_all(results, args):
    regrets = []
    for entry in results['5block']:
        tower, reward, max_reward, stable, n_success = entry
        if not stable:
            reward = 0
        
        regret = (max_reward - reward)/max_reward
        regrets.append(regret)

    return '%.2f' % np.mean(regrets)

def calc_reward_all(results, args):
    rewards = []
    for entry in results['5block']:
        tower, reward, max_reward, stable, n_success = entry
        if not stable:
            reward = 0
        rewards.append(reward)

    return '%.2f' % np.mean(rewards)

def calc_max_reward_all(results, args):
    rewards = []
    for entry in results['5block']:
        tower, reward, max_reward, stable, n_success = entry
        rewards.append(max_reward)

    return '%.2f' % np.mean(rewards)

def calc_n_stable(results, args):
    n_stable = 0
    n_total = 0
    for entry in results['5block']:
        tower, reward, max_reward, stable, n_success = entry
        n_stable += stable
        n_total += 1
    return f'{n_stable}/{n_total}'

def generate_table(args):
    logger = ActiveExperimentLogger(args.exp_path)

    metric_headings = {
        'reward-all': 'Avg. Reward',
        'max-reward-all': 'Max Avg. Reward',
        'regret-all': 'Avg. Regret',
        'reward-partial': 'Avg. Partial Reward',
        'regret-partial': 'Avg. Partial Regret',
        'reward-stable': 'Avg. Stable Reward',
        'regret-stable': 'Avg. Stable Regret',
        'n-stable': '# Stable'
    }

    metric_callbacks = {
        'reward-all': calc_reward_all,
        'max-reward-all': calc_max_reward_all,
        'regret-all': calc_regret_all,
        'reward-partial': calc_reward_partial,
        'regret-partial': calc_regret_partial,
        'reward-stable': calc_reward_stable,
        'regret-stable': calc_regret_stable,
        'n-stable': calc_n_stable
    }

    method_headings = {
        'learned': 'Learned Panda Robot Model',
        'simple-model': 'Analytical Constructability Model',
        'noisy-model': 'Simulation Model (5mm noise)'
    }

    n_metrics = len(args.metrics)
    table_latex = '\\begin{tabular}{|c|' + '|'.join(['c']*n_metrics) + '|}\n\\hline\n'
    table_latex += '\\textbf{Constructability Model} & ' + ' & '.join(['\\textbf{' + metric_headings[m] + '}' for m in args.metrics]) + ' \\\\\n'
    table_latex += '\\hline\n'

    for method in args.planning_models:
        table_latex += method_headings[method] 
        results = logger.get_evaluation_labels(args.problem, method, args.acquisition_step)
        results = list(results.values())[0]
    
        for metric in args.metrics:
            table_latex += ' & ' + metric_callbacks[metric](results, args) 
        table_latex += '\\\\\n'
        table_latex += '\\hline\n'

    table_latex += '\end{tabular}\n'

    return table_latex 
# Analytical Constructability Model & - & 3/10\\
# Simulation Model ($5$ mm noise)& - & \\ 
# Learned Panda Robot Model& - &  10/10\\ 
# \hline
# \end{tabular}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-path', 
                        type=str)
    parser.add_argument('--acquisition-step',
                        type=int,
                        help='acquisition step to evaluate (use either this or --max-acquisition)')
    parser.add_argument('--problem',
                        type=str,
                        default=['min-contact']) # tallest overhang cumulative-overhang
    parser.add_argument('--planning-models',
                        nargs='+',
                        type=str,
                        default=['simple-model', 'noisy-model', 'learned'])
    parser.add_argument('--metrics',
                        nargs='+',
                        type=str,
                        default=['reward-all', 'max-reward-all', 'regret-all', 'reward-partial', 'regret-partial', 'reward-stable', 'regret-stable', 'n-stable'])
    args = parser.parse_args()

    table = generate_table(args)
    print(table)