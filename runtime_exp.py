"""
File for obtaining the runtime of various OCP algorithms.
"""
import argparse
from collections import defaultdict
import math
import os
from re import sub

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, LBFGS, SGD
import tqdm

# Our new algorithm
from online_conformal.magnitude_learner import MagnitudeLearner, MagnitudeLearnerV2
from online_conformal.mag_learner_undiscounted import MagLearnUndiscounted
from online_conformal.ogd_simple import SimpleOGD

# From previous work
from online_conformal.saocp import SAOCP
from online_conformal.faci import FACI, FACI_S
from online_conformal.nex_conformal import NExConformal
from online_conformal.ogd import ScaleFreeOGD
from online_conformal.split_conformal import SplitConformal
from online_conformal.utils import pinball_loss
from cv_utils import create_model, data_loader
from cv_utils import ImageNet, TinyImageNet, CIFAR10, CIFAR100, ImageNetC, TinyImageNetC, CIFAR10C, CIFAR100C

import time

from helper_functions import *
__file__ = "runtime_exp.py"


# Train the model, save its logits on all the corrupted test datasets, and do temperature scaling
print("Train the model, save its logits on all the corrupted test datasets, and do temperature scaling")
args = parse_args()
# if args.dataset != "ImageNet":
if not finished(args) and args.dataset != "ImageNet":
    print("Training models")
    train(args)
    print("Finished training")
if args.local_rank in [-1, 0]:
    print("Getting temp file...")
    temp_file = get_temp_file(args)
    print("Done")
    if not finished(args):
        print("get_logits(args)")
        get_logits(args)
        print("...Done")
        temp = temperature_scaling(args)
        with open(temp_file, "w") as f:
            f.write(str(temp))

    # Load the saved logits
    print("Load the saved logits")
    with open(temp_file) as f:
        temp = float(f.readline())
    n_data = None
    sev2results = defaultdict(list)
    for corruption in corruptions:
        severities = [0] if corruption is None else [1, 2, 3, 4, 5]
        for severity in severities:
            try:
                logits, labels = torch.load(get_results_file(args, corruption, severity))
            except:
                continue
            sev2results[severity].append(list(zip(F.softmax(logits / temp, dim=-1).numpy(), labels.numpy())))
            n_data = len(labels) if n_data is None else min(n_data, len(labels))

    # Initialize conformal prediction methods, along with accumulators for results
    print("Initialize conformal prediction methods, along with accumulators for results")
    lmbda, k_reg, n_class = raps_params(args.dataset)
    D_old = 1 + lmbda * np.sqrt(n_class - k_reg)
    D = 1*D_old
    methods_iterate = [SimpleOGD, MagnitudeLearnerV2, SplitConformal, NExConformal, FACI, ScaleFreeOGD, FACI_S, SAOCP, MagnitudeLearner, MagLearnUndiscounted]
    #methods = [SplitConformal]
    label2err = defaultdict(list)
    h = 5 + 0.5 * (len(methods_iterate) > 5)
    np.random.seed(0)
    num_trials = 10
    elasped_time = np.zeros(num_trials,)
    time_simpleogd = np.zeros_like(elasped_time)
    mean_time_norm = np.zeros(len(methods_iterate))
    std_time_norm = np.zeros(len(methods_iterate))
    for method_ in methods_iterate:
        methods = [method_]
        i_method = methods_iterate.index(method_)
        print("Current method: ", method_.__name__)
        for jj in range(num_trials):
            for i_shift, shift in enumerate(["sudden"]):
                sevs, s_opts, w_opts = [], [], []
                # warmup, window, run_length = 1000, 100, 500 # original code
                warmup, window, run_length = 1000, 100, 1000 # our code
                state = np.random.RandomState(0)
                order = state.permutation(n_data)[: 6 * run_length + window // 2 + warmup]
                coverages, s_hats, widths = [{m.__name__: [] for m in methods} for _ in range(3)]
                predictors = [m(None, None, max_scale = D, lifetime = 32, coverage = args.target_cov) for m in methods]
                t_vec = np.zeros(len(order))
                start_time = time.time()
                for t, i in tqdm.tqdm(enumerate(order, start=-warmup), total=len(order)):
                    # Get saved results for the desired severity
                    sev = t_to_sev(t, window=window, schedule = shift)
                    probs, label = sev2results[sev][state.randint(0, len(sev2results[sev]))][i]

                    # Convert probability to APS score
                    i_sort = np.flip(np.argsort(probs))
                    p_sort_cumsum = np.cumsum(probs[i_sort]) - state.rand() * probs[i_sort]
                    s_sort_cumsum = p_sort_cumsum + lmbda * np.sqrt(np.cumsum([i > k_reg for i in range(n_class)]))
                    w_opt = np.argsort(i_sort)[label] + 1
                    s_opt = s_sort_cumsum[w_opt - 1]
                    if t >= 0:
                        sevs.append(sev)
                        s_opts.append(s_opt)
                        w_opts.append(w_opt)
                        t_vec[t] = t

                    # Update all the conformal predictors
                    for predictor in predictors:
                        name = type(predictor).__name__
                        if t >= 0:
                            _, s_hat = predictor.predict(horizon=1)
                            w = np.sum(s_sort_cumsum <= s_hat)
                            s_hats[name].append(s_hat)
                            widths[name].append(w)
                            coverages[name].append(w >= w_opt)
                        predictor.update(ground_truth=pd.Series([s_opt]), forecast=pd.Series([0]), horizon=1)
                #print("Total timesteps:", t)
                elasped_time[jj] = time.time() - start_time
                if method_.__name__ == "SimpleOGD":
                    time_simpleogd[jj] = elasped_time[jj]
                # Perform evaluation & produce a pretty graph
                plot_loss = False

                s_opts = np.asarray(s_opts)
                int_q = pd.Series(s_opts).rolling(window).quantile(args.target_cov).dropna()
                #print(f"Distribution shift: {shift}")
                if jj == num_trials-1:
                    for i, m in enumerate(methods):
                    # Compute various summary statistics
                        name = m.__name__ # name of the method (OGD, SAOCP, etc.)
                        label = sub("Split", "S", sub("Conformal", "CP", sub("ScaleFree", "SF-", sub("_", "-", name))))
                        s_hat = np.asarray(s_hats[name])
                        int_cov = gaussian_filter1d(pd.Series(coverages[name]).rolling(window).mean().dropna(), sigma=3)
                        int_w = pd.Series(s_hats[name] if plot_loss else widths[name]).rolling(window).mean().dropna()
                        int_losses = pd.Series(pinball_loss(s_opts, s_hat, args.target_cov)).rolling(window).mean().dropna()
                        opts = [pinball_loss(s_opts[i : i + window], q, args.target_cov).mean() for i, q in enumerate(int_q)]
                        int_regret = int_losses.values - np.asarray(opts)
                        int_miscov = np.abs(args.target_cov - int_cov)

                        # Do the plotting
                        label2err[label].append(f"{np.max(int_miscov):.2f}")
                        mean_time_simpleogd = np.mean(time_simpleogd)
                        mean_time_norm[i_method] = (np.mean(elasped_time))/mean_time_simpleogd
                        std_time_norm[i_method] = (np.std(elasped_time))/mean_time_simpleogd
                        print(
                            f"{name:15}: "
                            f"Cov: {np.mean(coverages[name]):.3f}, "
                            f"Avg Width: {np.mean(widths[name]):.1f}, "
                            f"Avg Miscov: {np.mean(int_miscov):.3f}, "
                            f"Avg Regret: {np.mean(int_regret):.4f}, "
                            f"Avg. Runtime (n=10): {mean_time_norm[i_method]:.6f}, "
                            f"Std. Runtime (n=10): {std_time_norm[i_method]:.6f}, "
                            f"LCEk: {np.max(int_miscov):.2f},"
                        )