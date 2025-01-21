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

from src.utils.data import get_loaders
from src.utils import name_match, metrics
from src.utils.utils import manual_designed_folder
from src.utils.early_stopping import EarlyStopper
from config.parser import Parser
import warnings

warnings.filterwarnings("ignore")


def main():
    parser = Parser()
    args = parser.parse()

    cf = lg.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch = lg.StreamHandler()

    result_path = "" + "result/"
    table_path = manual_designed_folder(result_path, args)
    if not os.path.exists(table_path):
        os.makedirs(table_path)

    for run_id in [0]:
        # Re-parse tag. Useful when using multiple runs.
        args = parser.parse()
        args.run_id = run_id

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

        # Dataloaders
        dataloaders = get_loaders(args)
        if args.training_type == 'inc':
            for task_id in range(args.n_tasks):
                task_name = f"train{task_id}"
                model_state = os.path.join(args.ckpt_root, f"{args.tag}/{args.run_id}/ckpt_train{task_id}.pth")
                mem_idx = int(len(dataloaders['train']) * args.batch_size / args.n_tasks) * (task_id + 1)
                buffer_state = os.path.join(args.ckpt_root, f"{args.tag}/{args.run_id}/memory_{mem_idx}.pkl")
                path = os.path.join('./checkpoints', args.dataset, args.learner, str(args.mem_size), str(args.seed))
                learner.resume(path)
                learner.before_eval()

                if args.learner == 'ERCCLDC' or args.learner == 'OnProCCLDC' or args.learner == 'OCMCCLDC' \
                        or args.learner == 'GSACCLDC' or args.learner == 'DERPPCCLDC' or args.learner == 'ERACECCLDC' or args.learner == 'SDAMCL':
                    if task_id == args.n_tasks - 1:
                        # learner.tsne_visualization(dataloaders, task_id)
                        learner.cam_visualization(dataloaders, task_id)
                else:
                    if task_id == args.n_tasks-1:
                        # learner.tsne_visualization(dataloaders, task_id)
                        learner.cam_visualization(dataloaders, task_id)

                learner.after_eval()


        # save model and buffer to "./checkpoints"
        # path = os.path.join('./checkpoints', args.dataset, args.learner, str(args.mem_size), str(args.seed))
        # learner.save(path)
    sys.exit(0)


if __name__ == '__main__':
    main()
