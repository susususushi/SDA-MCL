import pandas as pd
import numpy as np


from scipy.stats import sem
import scipy.stats as stats

def compute_performance(end_task_acc_arr):
    """
    Given test accuracy results from multiple runs saved in end_task_acc_arr,
    compute the average accuracy, forgetting, and task accuracies as well as their confidence intervals.

    :param end_task_acc_arr:       (list) List of lists
    :param task_ids:                (list or tuple) Task ids to keep track of
    :return:                        (avg_end_acc, forgetting, avg_acc_task)
    """
    n_run, n_tasks = end_task_acc_arr.shape[:2]
    t_coef = stats.t.ppf((1+0.95) / 2, n_run-1)     # t coefficient used to compute 95% CIs: mean +- t *

    # compute average test accuracy and CI
    end_acc = end_task_acc_arr[:, -1, :]                         # shape: (num_run, num_task)
    avg_acc_per_run = np.nanmean(end_acc, axis=1)      # mean of end task accuracies per run
    avg_end_acc = (np.nanmean(avg_acc_per_run), t_coef * sem(avg_acc_per_run))

    # compute forgetting
    best_acc = np.nanmax(end_task_acc_arr, axis=1)
    final_forgets = best_acc - end_acc
    avg_fgt = np.nanmean(final_forgets, axis=1)
    avg_end_fgt = (np.nanmean(avg_fgt), t_coef * sem(avg_fgt))

    # compute ACC
    acc_per_run = np.nanmean((np.nansum(np.tril(end_task_acc_arr), axis=2) /
                           (np.arange(n_tasks) + 1)), axis=1)
    avg_acc = (np.nanmean(acc_per_run), t_coef * sem(acc_per_run))


    # compute BWT+
    bwt_per_run = (np.nansum(np.tril(end_task_acc_arr, -1), axis=(1,2)) - np.nansum(np.diagonal(end_task_acc_arr, axis1=1, axis2=2) * (np.arange(n_tasks, 0, -1) - 1), axis=1)) / (n_tasks * (n_tasks - 1) / 2)
    bwtp_per_run = np.maximum(bwt_per_run, 0)
    avg_bwtp = (np.nanmean(bwtp_per_run), t_coef * sem(bwtp_per_run))

    # compute FWT
    fwt_per_run = np.nansum(np.triu(end_task_acc_arr, 1), axis=(1,2)) / (n_tasks * (n_tasks - 1) / 2)
    avg_fwt = (np.nanmean(fwt_per_run), t_coef * sem(fwt_per_run))

    # compute LA
    la_per_run = np.nanmean(np.diagonal(end_task_acc_arr, axis1=1, axis2=2), axis=1)
    avg_la = (np.nanmean(la_per_run), t_coef * sem(la_per_run))

    return avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt, avg_la



def single_run_avg_end_fgt(acc_array):
    best_acc = np.max(acc_array, axis=1)
    end_acc = acc_array[-1]
    final_forgets = best_acc - end_acc
    avg_fgt = np.mean(final_forgets)
    return avg_fgt


def forgetting_table(acc_dataframe, n_tasks=5):
    return pd.DataFrame(    
        [forgetting_line(acc_dataframe, task_id=i).values[:,-1].tolist() for i in range(0, n_tasks)]
    )

def forgetting_line(acc_dataframe, task_id=4, n_tasks=5):
    if task_id == 0:
        forgettings = [np.nan] * n_tasks
    else:
        forgettings = [forgetting(task_id, p, acc_dataframe) for p in range(task_id)] + [np.nan]*(n_tasks-task_id)

    # Create dataframe to handle NaN
    return pd.DataFrame(forgettings)

def forgetting(q, p, df):
    D = {}
    for i in range(0, q+1):
        D[f"d{i}"] = df.diff(periods=-i)

    # Create datafrmae to handle NaN
    return pd.DataFrame(([D[f'd{k}'].iloc[q-k,p] for k in range(0, q+1)])).max()[0]