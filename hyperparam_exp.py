"""
File for testing the effects of the estimated magnitude of the prediction radius for the online conformal prediction algorithms.
"""
from collections import defaultdict
from re import sub
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import torch
import torch.nn.functional as F
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

# Import helper functions
from helper_functions import *

args = parse_args()
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
    D_true = 1 + lmbda * np.sqrt(n_class - k_reg)
    D_list = np.array([1e-3, 1e-2, 1e-1, 1, 10, 100, 100])*D_true
    #methods = [SimpleOGD, MagnitudeLearner, MagLearnUndiscounted, MagnitudeLearnerV2]
    methods = [SimpleOGD, ScaleFreeOGD, SAOCP, SplitConformal, NExConformal, FACI, FACI_S, MagnitudeLearner, MagLearnUndiscounted, MagnitudeLearnerV2]
        
    avg_cov = np.zeros((len(D_list), len(methods)))
    avg_width = np.zeros_like(avg_cov)
    lce_k = np.zeros_like(avg_cov)

    for i_D in range(len(D_list)):
        D = D_list[i_D]
        label2err = defaultdict(list)
        h = 5 + 0.5 * (len(methods) > 5)
        np.random.seed(0)
        for i_shift, shift in enumerate(["sudden"]):
            
            sevs, s_opts, w_opts = [], [], []
            # warmup, window, run_length = 1000, 100, 500 # original code
            warmup, window, run_length = 1000, 100, 1000 # our code
            state = np.random.RandomState(0)
            order = state.permutation(n_data)[: 6 * run_length + window // 2 + warmup]
            coverages, s_hats, widths = [{m.__name__: [] for m in methods} for _ in range(3)]
            predictors = [m(None, None, max_scale = D, lifetime = 32, coverage = args.target_cov) for m in methods]
            t_vec = np.zeros(len(order))
            for t, i in tqdm.tqdm(enumerate(order, start=-warmup), total=len(order)):
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

            plot_loss = False
            sevs = pd.Series(sevs).rolling(window).mean().dropna()
            w_opts = pd.Series(s_opts if plot_loss else w_opts).rolling(window).quantile(args.target_cov).dropna()

            s_opts = np.asarray(s_opts)
            int_q = pd.Series(s_opts).rolling(window).quantile(args.target_cov).dropna()
            print(f"Distribution shift: {shift}")
            for i, m in enumerate(methods):
                # Compute various summary statistics
                name = m.__name__ # name of the method (OGD, SAOCP, etc.)
                label = sub("Split", "S", sub("Conformal", "CP", sub("ScaleFree", "SF-", sub("_", "-", name))))
                if name == "MagnitudeLearner":
                    label = "MagL-D"
                if name == "MagLearnUndiscounted":
                    label = "MagL"
                if name == "MagnitudeLearnerV2":
                    label = "MagDis"
                s_hat = np.asarray(s_hats[name])
                int_cov = gaussian_filter1d(pd.Series(coverages[name]).rolling(window).mean().dropna(), sigma=3)
                int_w = pd.Series(s_hats[name] if plot_loss else widths[name]).rolling(window).mean().dropna()
                int_losses = pd.Series(pinball_loss(s_opts, s_hat, args.target_cov)).rolling(window).mean().dropna()
                opts = [pinball_loss(s_opts[i : i + window], q, args.target_cov).mean() for i, q in enumerate(int_q)]
                int_regret = int_losses.values - np.asarray(opts)
                int_miscov = np.abs(args.target_cov - int_cov)

                # Average values for plotting
                avg_cov[i_D, i] = np.mean(coverages[name])
                avg_width[i_D, i] = np.mean(widths[name])
                lce_k[i_D, i] = np.max(int_miscov)
                #
                print(
                    f"{name:15} :, "
                    f"D_ratio = {D/D_true} ",
                    f"Avg Cov: {avg_cov[i_D,i]:.3f}, "
                    f"Avg Width: {avg_width[i_D,i]:.1f}, "
                    f"LCEk: {lce_k[i_D,i]:.2f},"
                )

# Save the results
np.savez_compressed('hyperparam_results.npz',methods=methods,D_list=D_list,avg_cov=avg_cov, avg_width=avg_width, lce_k=lce_k)