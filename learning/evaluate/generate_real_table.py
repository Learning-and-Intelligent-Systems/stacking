import argparse
import numpy as np
from learning.active.utils import ActiveExperimentLogger

task_headings = {
    'min-contact': 'Max Unsupported Area',
    'cumulative-overhang': 'Longest Overhang',
    'tallest': 'Tallest Tower'
}

metric_headings = {
    'reward-all': 'Avg. Reward',
    'max-reward-all': 'Max Avg. Reward',
    'regret-all': 'Regret',
    'reward-partial': 'Avg. Partial Reward',
    'regret-partial': 'Avg. Partial Regret',
    'reward-stable': 'Avg. Stable Reward',
    'regret-stable': 'Stable Regret',
    'n-stable': '\\# Stable'
}


method_headings = {
    'learned': 'Learned Panda Model',
    'simple-model': 'Analytical',
    'noisy-model': 'Simulation (5mm noise)'
}

def cumulative_overhang(tower):
    total_overhang = 0
    for tx in range(1, tower.shape[0]):
        bx = tx - 1
        overhang = (tower[tx, 8] + tower[tx, 5]/2.) - (tower[bx, 8] + tower[bx, 5]/2.)
        total_overhang += overhang
    return total_overhang

def tower_height(tower):
    """
    :param tower: A vectorized version of the tower.
    """
    return np.sum(tower[:, 6])

def calc_reward_stable(results, task):
    rewards = []
    for entry in results['5block']:
        tower, reward, max_reward, stable, n_success = entry
        if not stable:
            reward = 0
        if stable:
            rewards.append(reward)

    return '%.2f' % np.mean(rewards)

def calc_regret_stable(results, task):
    regrets = []
    for entry in results['5block']:
        tower, reward, max_reward, stable, n_success = entry
        if not stable:
            reward = 0
        
        if stable:
            regret = (max_reward - reward)/max_reward
            regrets.append(regret)

    return '%.2f' % np.mean(regrets)

def calc_reward_partial(results, task):
    rewards = []
    for entry in results['5block']:
        tower, reward, max_reward, stable, n_success = entry
        tower_vec = np.array([b.vectorize() for b in tower[:n_success]])
        if task == 'cumulative-overhang':
            reward = cumulative_overhang(tower_vec)
        elif task == 'tallest':
            reward = tower_height(tower_vec)
        else:
            raise NotImplementedError()
        rewards.append(reward)

    return '%.2f' % np.mean(rewards)

def calc_regret_partial(results, task):
    regrets = []
    for entry in results['5block']:
        tower, reward, max_reward, stable, n_success = entry
        tower_vec = np.array([b.vectorize() for b in tower[:n_success]])
        if task == 'cumulative-overhang':
            reward = cumulative_overhang(tower_vec)
        elif task == 'tallest':
            reward = tower_height(tower_vec)
        else:
            raise NotImplementedError()
        regret = (max_reward - reward)/max_reward
        regrets.append(regret)

    return '%.2f' % np.mean(regrets)

def calc_regret_all(results, task):
    regrets = []
    for entry in results['5block']:
        tower, reward, max_reward, stable, n_success = entry
        if not stable:
            reward = 0
        
        regret = (max_reward - reward)/max_reward
        regrets.append(regret)

    return '%.2f' % np.mean(regrets)

def calc_reward_all(results, task):
    rewards = []
    for entry in results['5block']:
        tower, reward, max_reward, stable, n_success = entry
        if not stable:
            reward = 0
        rewards.append(reward)

    return '%.2f' % np.mean(rewards)

def calc_max_reward_all(results, task):
    rewards = []
    for entry in results['5block']:
        tower, reward, max_reward, stable, n_success = entry
        rewards.append(max_reward)

    return '%.2f' % np.mean(rewards)

def calc_n_stable(results, task):
    n_stable = 0
    n_total = 0
    for entry in results['5block']:
        tower, reward, max_reward, stable, n_success = entry
        n_stable += stable
        n_total += 1
    return f'{n_stable}/{n_total}'


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

def generate_single_table(args):
    logger = ActiveExperimentLogger(args.exp_path)

    n_metrics = len(args.metrics)
    table_latex = '\\begin{tabular}{|c|' + '|'.join(['c']*n_metrics) + '|}\n\\hline\n'
    table_latex += '\\textbf{APF Model} & ' + ' & '.join(['\\textbf{' + metric_headings[m] + '}' for m in args.metrics]) + ' \\\\\n'
    table_latex += '\\hline\n'

    for method in args.planning_models:
        table_latex += method_headings[method] 
        results = logger.get_evaluation_labels(args.problems[0], method, args.acquisition_step)
        results = list(results.values())[0]
    
        for metric in args.metrics:
            table_latex += ' & ' + metric_callbacks[metric](results, args.problem) 
        table_latex += '\\\\\n'
        table_latex += '\\hline\n'

    table_latex += '\end{tabular}\n'

    return table_latex 

def generate_multi_table(args):
    logger = ActiveExperimentLogger(args.exp_path)

    n_metrics = len(args.metrics)
    n_tasks = len(args.problems)
    table_latex = '\\begin{tabular}{|c|'
    # Add the appropriate number of columns: n_tasks*n_metrics
    for _ in range(n_tasks):
        single_task = '|' + '|'.join(['c']*n_metrics) + '|'
        table_latex += single_task
    table_latex += '}\n\\hline\n'

    # Add a title for each task.
    for tx, task in enumerate(args.problems):
        if tx == len(args.problems) - 1:
            task_name = ' & \\multicolumn{' + str(n_metrics) + '}{c|}{\\textbf{' + task_headings[task] + '}}'
        else:
            task_name = ' & \\multicolumn{' + str(n_metrics) + '}{c||}{\\textbf{' + task_headings[task] + '}}'
        table_latex += task_name
    table_latex += '\\\\ \n\\hline\n'

    # For each task, add metric columns.
    table_latex += '\\textbf{APF Model} '
    for _ in range(n_tasks):
        table_latex += '& ' +  ' & '.join(['\\textbf{' + metric_headings[m] + '}' for m in args.metrics]) 
        
    table_latex += '\\\\ \\hline\n'

    for method in args.planning_models:
        table_latex += method_headings[method] 

        for task in args.problems:
            results = logger.get_evaluation_labels(task, method, args.acquisition_step)
            if len(results) > 0:
                results = list(results.values())[0]
            else:
                results = None
        
            for metric in args.metrics:
                if results is None:
                    table_latex += ' & '
                else:
                    table_latex += ' & ' + metric_callbacks[metric](results, task) 
                
        table_latex += '\\\\\n'
        table_latex += '\\hline\n'

    table_latex += '\end{tabular}\n'
    return table_latex 

    

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-path', 
                        type=str)
    parser.add_argument('--acquisition-step',
                        type=int,
                        help='acquisition step to evaluate (use either this or --max-acquisition)')
    parser.add_argument('--problems',
                        nargs='+',
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

    if len(args.problems) == 1:
        table = generate_single_table(args)
    else:
        table = generate_multi_table(args)
    print(table)