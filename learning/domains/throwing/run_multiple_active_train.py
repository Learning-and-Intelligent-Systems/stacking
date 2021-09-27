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
        elif isinstance(v, bool) and v == True: cmd += f" --{k.replace('_', '-')}"
        elif isinstance(v, bool) and v == False: continue
        else: cmd += f" --{k.replace('_', '-')}={v}"

    return cmd

def find_path_to_exp(prefix, exp_path="learning/experiments/logs"):
    """ find the first experiment in exp_path with a given prefix """
    for fname in os.listdir(exp_path):
        if fname.startswith(prefix):
            return exp_path + '/' + fname
    return None

def run_commands_with_subprocess(commands):
    """ execute a list of shell commands in parallel """
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

def run_with_subprocess(args, dry=False):
    """ take argparse namesspace for active_train and run multiple training
    runs in parallel """
    commands = []
    for i in range(args.n_runs):
        if not args.fitting:
            args.exp_name = args.prefix + f"_run_{i}"

        else:
            args.exp_name = args.prefix + f"_fitting_run_{i}"
            train_exp_path = find_path_to_exp(args.prefix + f"_run_{i}")
            if train_exp_path is not None:
                args.latent_ensemble_exp_path = train_exp_path
            else:
                print("Failed to find experiment with prefix", args.prefix)
                continue

        # generate the CLI command
        cmd = args_to_command("python -m learning.domains.throwing.active_train",
            args, skip_args=["prefix", "n_runs", "dry"])
        # print it (for dry run viewing)
        print(cmd)
        commands.append(cmd)

    if not dry:
        run_commands_with_subprocess(commands)

def run_both_phases(args, dry=False):
    """ run both the training and fitting phases """
    run_with_subprocess(args, dry=dry)
    print("[TRAINING COMPLETE]")

    args.fitting = True
    args.latent_ensemble_tx = args.max_acquisitions
    args.n_objects = 1
    run_with_subprocess(args, dry=dry)
    print("[FITTING COMPLETE]")


def run_manually():
    commands = [
    "python -m learning.domains.throwing.active_train --max-acquisitions=100 --exp-name=throwing_20_objects_random_then_bald_fitting_run_0 --batch-size=16 --n-models=10 --n-latent-samples=10 --n-epochs=150 --n-samples=1000 --n-acquire=5 --n-objects=1 --hide-dims=0,1,2,3,4,5,6,7,8,9 --acquisition=bald --object-fname= --use-latents --use-normalization --fitting --latent-ensemble-exp-path=learning/experiments/logs/sept_24th/throwing_20_objects_random_run_0-20210924-110249 --latent-ensemble-tx=100",
    "python -m learning.domains.throwing.active_train --max-acquisitions=100 --exp-name=throwing_20_objects_random_then_bald_fitting_run_1 --batch-size=16 --n-models=10 --n-latent-samples=10 --n-epochs=150 --n-samples=1000 --n-acquire=5 --n-objects=1 --hide-dims=0,1,2,3,4,5,6,7,8,9 --acquisition=bald --object-fname= --use-latents --use-normalization --fitting --latent-ensemble-exp-path=learning/experiments/logs/sept_24th/throwing_20_objects_random_run_1-20210924-110250 --latent-ensemble-tx=100",
    "python -m learning.domains.throwing.active_train --max-acquisitions=100 --exp-name=throwing_20_objects_random_then_bald_fitting_run_2 --batch-size=16 --n-models=10 --n-latent-samples=10 --n-epochs=150 --n-samples=1000 --n-acquire=5 --n-objects=1 --hide-dims=0,1,2,3,4,5,6,7,8,9 --acquisition=bald --object-fname= --use-latents --use-normalization --fitting --latent-ensemble-exp-path=learning/experiments/logs/sept_24th/throwing_20_objects_random_run_2-20210924-110249 --latent-ensemble-tx=100",
    "python -m learning.domains.throwing.active_train --max-acquisitions=100 --exp-name=throwing_20_objects_random_then_bald_fitting_run_3 --batch-size=16 --n-models=10 --n-latent-samples=10 --n-epochs=150 --n-samples=1000 --n-acquire=5 --n-objects=1 --hide-dims=0,1,2,3,4,5,6,7,8,9 --acquisition=bald --object-fname= --use-latents --use-normalization --fitting --latent-ensemble-exp-path=learning/experiments/logs/sept_24th/throwing_20_objects_random_run_3-20210924-110249 --latent-ensemble-tx=100",
    "python -m learning.domains.throwing.active_train --max-acquisitions=100 --exp-name=throwing_20_objects_random_then_bald_fitting_run_4 --batch-size=16 --n-models=10 --n-latent-samples=10 --n-epochs=150 --n-samples=1000 --n-acquire=5 --n-objects=1 --hide-dims=0,1,2,3,4,5,6,7,8,9 --acquisition=bald --object-fname= --use-latents --use-normalization --fitting --latent-ensemble-exp-path=learning/experiments/logs/sept_24th/throwing_20_objects_random_run_4-20210924-110249 --latent-ensemble-tx=100"
    ]

    print("\n".join(commands))
    run_commands_with_subprocess(commands)


if __name__ == '__main__':
    run_manually()
    # parser = get_parser() # parser for active_train
    # # additional arguments for running multiple experiments
    # parser.add_argument('--n-runs', type=int, default=5)
    # parser.add_argument('--prefix', type=str, default='throwing')
    # parser.add_argument('--dry', action='store_true')
    # args = parser.parse_args()

    # # run_with_subprocess(args, dry=args.dry) # run a single phase
    # run_both_phases(args, dry=args.dry) # run both phases
