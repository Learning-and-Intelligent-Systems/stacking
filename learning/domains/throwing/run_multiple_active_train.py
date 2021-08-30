from multiprocessing import Process
import numpy as np
import torch

from learning.domains.throwing.active_train import get_parser, run_active_throwing

if __name__ == '__main__':
    n_runs = 5

    args = get_parser().parse_args()
    args.use_latents = True
    args.n_objects = 10
    args.n_epochs = 100
    args.max_acquisitions = 50
    args.acquisition = "bald"

    processes = []
    for n in range(n_runs):
        args.exp_name = f'throwing_bald_sweep_run_{n}'
        np.random.seed() # this is critical!
        torch.random.seed() # this is critical!
        processes.append(Process(target=run_active_throwing, args=(args,)))
        np.random.seed() # this is critical!
        torch.random.seed() # this is critical!
        processes[-1].start()

    for p in processes:
        p.join()