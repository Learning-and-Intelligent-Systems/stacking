from multiprocessing import Process
from subprocess import Popen
import numpy as np
import os
import torch
import time

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


def run_with_multiprocessing():


    args.use_latents = True
    args.n_objects = 1
    args.n_epochs = 500
    args.max_acquisitions = 50
    args.hide_dims="9"
    args.latent_ensemble_tx = args.max_acquisitions - 1

    processes = []
    for n in range(args.n_runs):
        args.exp_name = args.prefix + f"_run_{n}"

        if args.fitting:
            # if we're fitting, we need to find the corresponding training run
            # and change the name of the experiment log to say "fitting"
            args.latent_ensemble_exp_path = find_experiment_path_by_name(args.exp_name)
            args.exp_name = args.prefix + f"_fitting_run_{n}"
            args.n_acquire = 1

        np.random.seed() # this is critical!
        torch.random.seed() # this is critical!
        processes.append(Process(target=run_active_throwing, args=(args,)))
        np.random.seed() # this is critical!
        torch.random.seed() # this is critical!
        processes[-1].start()

    for p in processes:
        p.join()


def args_to_command(cmd, args, skip_args=[]):
    """ takes an argparse namespace and converts it to a string for calling
    with os.system or subprocess. args in skip_args are left out """

    for k, v in args.__dict__.items():
        if k in skip_args: continue
        elif v == True: cmd += f" --{k.replace('_', '-')}"
        elif v == False: continue
        else: cmd += f" --{k.replace('_', '-')}={v}"

    return cmd

def run_with_subprocess(args, dry=False):
    commands = []
    for i in range(args.n_runs):
        # set the experiment logging directory name
        args.exp_name = args.prefix + f"_run_{i}"
        # generate the CLI command
        cmd = args_to_command("python -m learning.domains.throwing.active_train",
            args, skip_args=["prefix", "n_runs", "dry"])
        # print it (for dry run viewing)
        print(cmd)
        commands.append(cmd)

    if not dry:
        processes = [Popen(cmd, shell=True) for cmd in commands]

        try:
            while True:
                # poll the processes. break when all are done.
                time.sleep(1)
                if np.array([p.poll() == None for p in processes]).any():
                    continue
                else:
                    break

        except KeyboardInterrupt:
            print("[TERMINATING ALL EXPERIMENTS]")
            for p in processes:
                p.kill()


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('--n-runs', type=int, default=5)
    parser.add_argument('--prefix', type=str, default='throwing')
    parser.add_argument('--dry', action='store_true')
    args = parser.parse_args()

    run_with_subprocess(args, dry=args.dry)
    # print(args_to_command(args))