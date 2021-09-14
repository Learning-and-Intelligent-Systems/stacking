from multiprocessing import Process
import numpy as np
import os
import torch

from learning.domains.throwing.active_train import get_parser, run_active_throwing


def find_experiment_path_by_name(exp_name,
    path_to_experiments="learning/experiments/logs"):

    exps = [p for p in os.listdir(path_to_experiments) if p.startswith(exp_name)]
    if len(exps) != 1:
        print(f"[WARNING] Found {len(exps)} experiments with name, {exp_name}")
    if len(exps) > 0:
        return path_to_experiments + '/' + exps[0]
    else:
        return ""


if __name__ == '__main__':
    

    parser = get_parser()
    parser.add_argument('--n-runs', type=int, default=5)
    parser.add_argument('--sweep-name', type=str, default='sweep')
    args = parser.parse_args()

    args.use_latents = True
    args.n_objects = 1
    args.n_epochs = 500
    args.max_acquisitions = 50
    args.hide_dims="9"
    args.latent_ensemble_tx = args.max_acquisitions - 1

    processes = []
    for n in range(args.n_runs):
        args.exp_name = args.sweep_name + f"_run_{n}"

        if args.fitting:
            # if we're fitting, we need to find the corresponding training run
            # and change the name of the experiment log to say "fitting"
            args.latent_ensemble_exp_path = find_experiment_path_by_name(args.exp_name)
            args.exp_name = args.sweep_name + f"_fitting_run_{n}"
            args.n_acquire = 1

        np.random.seed() # this is critical!
        torch.random.seed() # this is critical!
        processes.append(Process(target=run_active_throwing, args=(args,)))
        np.random.seed() # this is critical!
        torch.random.seed() # this is critical!
        processes[-1].start()

    for p in processes:
        p.join()