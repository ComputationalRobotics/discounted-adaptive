"""
File containing helper functions that are used in the experiments for online conformal prediction.
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

def parse_args():
    args = argparse.Namespace
    args.dataset = "TinyImageNet"
    args.model = "resnet50"
    args.lr = 1e-3
    args.batch_size = 256
    args.n_epochs = 150 #default = 150
    args.patience = 10
    #args.ignore_checkpoint = "store_true"
    args.target_cov = 90
    args.ignore_checkpoint = True

    assert 50 < args.target_cov < 100
    args.target_cov = args.target_cov / 100

    # Set up distributed training if desired, and set the device
    args.local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if args.local_rank == -1:
        if torch.cuda.is_available():
            args.device = torch.device("cuda")
        else:
            args.device = torch.device("cpu")
        args.world_size = 1
    else:
        dist.init_process_group(backend="nccl")
        args.device = torch.device(args.local_rank)
        args.world_size = dist.get_world_size()
        
    return args

corruptions = [
    None,
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    "fog",
    "frost",
    "gaussian_noise",
    "glass_blur",
    "impulse_noise",
    "jpeg_compression",
    "motion_blur",
    "pixelate",
    "shot_noise",
    "snow",
    "zoom_blur",
]

__file__ = "vision.py"

def get_base_dataset(dataset, split):
    if dataset == "ImageNet":
        return ImageNet(split)
    elif dataset == "TinyImageNet":
        return TinyImageNet(split)
    elif dataset == "CIFAR10":
        return CIFAR10(split)
    elif dataset == "CIFAR100":
        return CIFAR100(split)
    raise ValueError(f"Dataset {dataset} is not supported.")


def get_model_file(args):
    rootdir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(rootdir, "cv_models", args.dataset, args.model, "model.pt")


def get_model(args):
    if args.dataset != "ImageNet":
        return torch.load(get_model_file(args), map_location=args.device)
    return create_model(dataset=ImageNet("valid"), model_name=args.model, device=args.device)


def get_results_file(args, corruption, severity):
    rootdir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(rootdir, "cv_logits", args.dataset, args.model, f"{corruption}_{severity}.pt")


def get_temp_file(args):
    return os.path.join(os.path.dirname(get_results_file(args, None, 0)), "temp.txt")


def finished(args):
    for corruption in corruptions:
        for severity in [0] if corruption is None else [1, 2, 3, 4, 5]:
            fname = get_results_file(args, corruption, severity)
            if not os.path.isfile(fname):
                return False
    return os.path.isfile(get_temp_file(args))


def raps_params(dataset):
    if dataset == "CIFAR10":
        lmbda, k_reg, n_class = 0.1, 1, 10
    elif dataset == "CIFAR100":
        lmbda, k_reg, n_class = 0.02, 5, 100
    elif dataset == "TinyImageNet":
        lmbda, k_reg, n_class = 0.01, 20, 200
    elif dataset == "ImageNet":
        lmbda, k_reg, n_class = 0.01, 10, 1000
    else:
        raise ValueError(f"Unsupported dataset {dataset}")
    return lmbda, k_reg, n_class


def temperature_scaling(args):
    temp = nn.Parameter(torch.tensor(1.0, device=args.device))
    opt = LBFGS([temp], lr=0.01, max_iter=500)
    loss_fn = nn.CrossEntropyLoss()

    n_epochs = 10
    valid_data = get_base_dataset(args.dataset, "valid")
    model = get_model(args)
    for epoch in range(n_epochs):
        valid_loader = data_loader(valid_data, batch_size=args.batch_size, epoch=epoch)
        for x, y in tqdm.tqdm(valid_loader, desc=f"Calibration epoch {epoch + 1:2}/{n_epochs}", disable=False):
            with torch.no_grad():
                logits = model(x.to(device=args.device))

            def eval():
                opt.zero_grad()
                loss = loss_fn(logits / temp, y.to(device=args.device))
                loss.backward()
                return loss

            opt.step(eval)

    return temp.item()


def get_logits(args):
    if args.dataset == "CIFAR10":
        dataset_cls = CIFAR10C
    elif args.dataset == "CIFAR100":
        dataset_cls = CIFAR100C
    elif args.dataset == "TinyImageNet":
        dataset_cls = TinyImageNetC
    elif args.dataset == "ImageNet":
        dataset_cls = ImageNetC
    else:
        raise ValueError(f"Dataset {args.dataset} is not supported.")
    model = None
    print("Dataset: ", dataset_cls)

    print("Applying corruptions: ", corruptions)
    for corruption in tqdm.tqdm(corruptions, desc="Corruptions", position=1):
        print("Corruption: ", corruption)
        severities = [0] if corruption is None else [1, 2, 3, 4, 5]
        print("Applying various severity levels: ", severities)
        for severity in tqdm.tqdm(severities, desc="Severity Levels", position=2, leave=False):
            print("Severity: ", severity)
            fname = get_results_file(args, corruption, severity)
            if os.path.isfile(fname) and not args.ignore_checkpoint:
                continue
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            if model is None:
                model = get_model(args)

            # Save the model's logits & labels for the whole dataset
            print("Save the model's logits & labels for the whole dataset")
            logits, labels = [], []
            dataset = dataset_cls(corruption=corruption, severity=severity)
            loader = data_loader(dataset, batch_size=args.batch_size)
            with torch.no_grad():
                for x, y in loader:
                    logits.append(model(x.to(device=args.device)).cpu())
                    labels.append(y.cpu())
            torch.save([torch.cat(logits), torch.cat(labels)], fname)


def t_to_sev(t, window, run_length=500, schedule=None):
    if t < window or schedule in [None, "None", "none"]:
        return 0
    t_base = t - window // 2
    if schedule == "gradual":
        k = (t_base // run_length) % 10
        return k if k <= 5 else 10 - k
    if schedule == "random_sudden":
        return np.clip(np.random.randint(0, 10) * ((t_base // run_length) % 2),0,5)
    if schedule == "random_gradual":
        k = (((t_base* abs(np.random.uniform(1,1.5))) // run_length) % 10 ) 
        return (k if k <= 5 else 10 - k) * np.random.randint(1,2) 
    return 5 * ((t_base // run_length) % 2) # default: sudden schedule


def train(args):
    # Get train/valid data
    print("Getting training and validation data")
    train_data = get_base_dataset(args.dataset, "train")
    valid_data = get_base_dataset(args.dataset, "valid")

    # Load model checkpoint one has been saved. Otherwise, initialize everything from scratch.
    print("Load model checkpoint one has been saved. Otherwise, initialize everything from scratch.")
    model_file = get_model_file(args)
    ckpt_name = os.path.join(os.path.dirname(model_file), "checkpoint.pt")
    if os.path.isfile(ckpt_name) and not args.ignore_checkpoint:
        model, opt, epoch, best_epoch, best_valid_acc = torch.load(ckpt_name, map_location=args.device)
    else:
        # create save directory if needed
        print("Create a save directory if needed")
        if args.local_rank in [-1, 0]:
            os.makedirs(os.path.dirname(ckpt_name), exist_ok=True)
        model = create_model(dataset=train_data, model_name=args.model, device=args.device)
        if "ImageNet" in args.dataset:
            opt = SGD(model.parameters(), lr=0.1, momentum=0.9)
        else:
            opt = Adam(model.parameters(), lr=args.lr)
        epoch, best_epoch, best_valid_acc = 0, 0, 0.0

    # Set up distributed data parallel if applicable
    print("Set up distributed data parallel if applicable")
    writer = args.local_rank in [-1, 0]
    if args.local_rank != -1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.device])

    for epoch in range(epoch, args.n_epochs):
        # Check early stopping condition
        print("Check early stopping condition")
        if args.patience and epoch - best_epoch > args.patience:
            break

        # Main training loop
        print("Main training loop")
        train_loader = data_loader(dataset=train_data, batch_size=args.batch_size // args.world_size, epoch=epoch)
        for x, y in tqdm.tqdm(train_loader, desc=f"Train epoch {epoch+1:2}/{args.n_epochs}", disable=not writer):
            opt.zero_grad()
            pred = model(x.to(device=args.device))
            loss = F.cross_entropy(pred, y.to(device=args.device))
            loss.backward()
            opt.step()

        # Anneal learning rate by a factor of 10 every 7 epochs
        print("Anneal learning rate by a factor of 10 every 7 epochs")
        if (epoch + 1) % 7 == 0:
            for g in opt.param_groups:
                g["lr"] *= 0.1

        # Obtain accuracy on the validation dataset
        print("Obtain accuracy on the validation dataset")
        valid_acc = torch.zeros(2, device=args.device)
        valid_loader = data_loader(valid_data, batch_size=args.batch_size, epoch=epoch)
        with torch.no_grad():
            for x, y in tqdm.tqdm(valid_loader, desc=f"Valid epoch {epoch + 1:2}/{args.n_epochs}", disable=True):
                pred = model(x.to(device=args.device))
                valid_acc[0] += x.shape[0]
                valid_acc[1] += (pred.argmax(dim=-1) == y.to(device=args.device)).sum().item()

        # Reduce results from all parallel processes
        print("Reduce results from all parallel processes")
        if args.local_rank != -1:
            dist.all_reduce(valid_acc)
        valid_acc = (valid_acc[1] / valid_acc[0]).item()

        # Save checkpoint & update best saved model
        print("Save checkpoints and update best saved model")
        if writer:
            print(f"Epoch {epoch + 1:2} valid acc: {valid_acc:.5f}")
            model_to_save = model.module if args.local_rank != -1 else model
            if valid_acc > best_valid_acc:
                best_epoch = epoch
                best_valid_acc = valid_acc
                torch.save(model_to_save, model_file)
            torch.save([model_to_save, opt, epoch + 1, best_epoch, best_valid_acc], ckpt_name)

        # Synchronize before starting next epoch
        print("Synchronize before starting next epoch")
        if args.local_rank != -1:
            dist.barrier()