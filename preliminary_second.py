import os
import torch
import pandas as pd
import numpy as np
import sys
import logging as lg
import datetime as dt
import random as r
import ssl
import wandb
import time

ssl._create_default_https_context = ssl._create_unverified_context
import matplotlib.pyplot as plt
from src.utils.datapre import get_loaders
from src.utils import name_match, metrics
from src.utils.utils import manual_designed_folder
from src.utils.early_stopping import EarlyStopper
from config.parser import Parser
import warnings

warnings.filterwarnings("ignore")


def main():
    runs_accs = []
    runs_fgts = []
    runs_scores = []
    runs_first_term = []
    runs_second_term = []
    parser = Parser()
    args = parser.parse()

    cf = lg.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch = lg.StreamHandler()

    accuracy_list = []
    start = time.time()
    for run_id in range(args.start_seed, args.start_seed + args.n_runs):
        # Re-parse tag. Useful when using multiple runs.
        args = parser.parse()
        args.run_id = run_id

        if args.sweep:
            wandb.init()
            for key in wandb.config.keys():
                setattr(args, key, wandb.config[key])

        # Seed initilization
        if args.n_runs > 1:
            args.seed = run_id
        np.random.seed(args.seed)
        r.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available() and args.learner == 'OnProDML':
            torch.cuda.manual_seed(args.seed)
        elif torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Learner
        if args.learner is not None:
            learner = name_match.learners[args.learner](args)
            if args.resume: learner.resume(args.model_state, args.buffer_state)
        else:
            raise Warning("Please select the desired learner.")

        # logs
        # Define logger and timstamp
        logfile = f'{args.tag}.log'
        if not os.path.exists(args.logs_root): os.mkdir(args.logs_root)

        ff = lg.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        logger = lg.getLogger()
        fh = lg.FileHandler(os.path.join(args.logs_root, logfile))
        ch.setFormatter(cf)
        fh.setFormatter(ff)
        logger.addHandler(fh)
        logger.addHandler(ch)
        if args.verbose:
            logger.setLevel(lg.DEBUG)
            logger.warning("Running in VERBOSE MODE.")
        else:
            logger.setLevel(lg.INFO)

        lg.info("=" * 60)
        lg.info("=" * 20 + f"RUN N°{run_id} SEED {args.seed}" + "=" * 20)
        lg.info("=" * 60)
        lg.info("Parameters used for this training")
        lg.info("=" * 20)
        lg.info(args)

        # Dataloaders
        dataloaders = get_loaders(args)

        # wandb initilization
        if not args.no_wandb and not args.sweep:
            wandb.init(
                project=f"{args.learner}",
                config=args.__dict__
            )

        # Training
        # Class incremental training
        if args.training_type == 'inc':
            for task_id in [0]:
                train_task_name = f"train{task_id}"
                validation_task_name = f"validation{task_id}"
                val_task_name = f"val{task_id}"
                if args.train:
                    learner.train(
                        dataloader=dataloaders[validation_task_name],
                        task_name=validation_task_name,
                        task_id=task_id,
                        dataloaders=dataloaders
                    )
                    learner.evaluate_and_save_irreducible_loss(
                        dataloader=dataloaders[val_task_name],
                    )
                    # learner.train(
                    #     dataloader=dataloaders[train_task_name],
                    #     task_name=train_task_name,
                    #     task_id=task_id,
                    #     dataloaders=dataloaders
                    # )
                    # learner.evaluate_and_save_model_loss(
                    #     dataloader=dataloaders[val_task_name],
                    # )
                    # valid_reducible_loss = learner.reducible_loss[torch.isfinite(learner.reducible_loss)].cpu().numpy()
                    # first_term = learner.model_loss[torch.isfinite(learner.model_loss)].cpu().numpy()
                    second_term = learner.irreducible_losses[torch.isfinite(learner.irreducible_losses)].cpu().numpy()

                    # runs_scores.append(valid_reducible_loss)
                    # runs_first_term.append(first_term)
                    runs_second_term.append(second_term)
                    # print('first_term:', np.mean(first_term), 'second_term:', np.mean(second_term))

    scores_table_path = "" + "scores/"
    scores_table_path = scores_table_path + args.dataset + "_mem=" + str(args.mem_size) + '/'
    scores_table_path = scores_table_path + args.augmentation + "_iters=" + str(args.mem_iters) + '/'
    if not os.path.exists(scores_table_path):
        os.makedirs(scores_table_path)
    score_path = scores_table_path + "second_term.npy"

    concatenated_second_term = np.concatenate(runs_second_term, axis=0)
    np.save(score_path, concatenated_second_term)

    # Exits the program
    sys.exit(0)


if __name__ == '__main__':
    main()
