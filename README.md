# Discounted Adaptive Online Prediction

For presentation at the 2024 International Conference on Machine Learning (ICML).

[![arXiv](https://img.shields.io/badge/arXiv-2402.02720-b31b1b.svg)](https://arxiv.org/abs/2402.02720)

## Citation

Zhiyu Zhang, David Bombara, and Heng Yang. Discounted adaptive online prediction. *arXiv preprint arXiv:2402.02720*, 2024.

### Bibtex Format

```text
@article{zhang2024discounted,
  title={Discounted Adaptive Online Prediction},
  author={Zhang, Zhiyu and Bombara, David and Yang, Heng},
  journal={arXiv preprint arXiv:2402.02720},
  year={2024}
}
```

## Getting Started

1. Create a virtual environment: Enter the following into the terminal at the project's root directory: 

    `python3 -m venv env`
2. Activate the environment:

    `source env/bin/activate`

3. Install necessary packages:

    `pip install -r requirements.txt`

## Training the Base Model

1. After getting started, the base image classification model must be trained on the user's machine. The models and logits will be automatically saved to the folders `cv_models/` and `cv_logits/`, respectively. This is the most time-consuming portion, but only must occur once. Replicating the experiments is quick in comparison.

## Replicating Experiments

1. Figure 1: To generate the data and plot, run `main_exp.py`. This will save the plot to `figures/TinyImageNet_1.0D.pdf`.
2. Figure 2: 
    - To generate the data, run `hyperparam_exp.py`, which will save data to `hyperparam_results.npz`.
    - To generate the plot, run `hyperparam_plot.py`, which will show the plot and save it to `figures/TinyImageNet_RadiusPrediction.pdf`.
3. Figure 3:
    - To generate the data, run `runtime_exp.py`, which will save data to `runtime_results.npz`.
    - To generate the plot, run `runtime_plot.py`, which will show the plot and save it to `figures/TinyImageNet_Runtime.pdf`. 
4. Table I: 
    - To generate the data and print results, run `benchmark_exp.py`. This will also generate a plot similar to Figure 1 containing the performance of 10 different algorithms.

## Algorithm Details

1. `online_conformal/magnitude_learner.py` contains the following:
    - **MagL-D** (**Mag**nitude Learner with **D**iscounting and **L**ipshitz Constant Estimate): Our Algorithm 1 from the manuscript with $\varepsilon = 1$ and discount factor $\lambda_t = 0.999$.
    - **MagDis** (**Mag**nitude Learner with **Dis**counting): Our Algorithm 5 from the manuscript with $\varepsilon = 1$ and $\lambda_t=0.999$, which is a simplified version of Algorithm 5 that essentially sets $h_t = 0$, does not clip $g_t$, and initializes $v_t > 0$.

2. `online_conformal/mag_learner_undiscounted.py` contains:
    - **MagL** (**Mag**nitude Learner with **L**ipshitz constant estimate): Our Algorithm 2, the undiscounted algorithm that uses the running estimate of the Lipshitz constant, $h_t$, with $\varepsilon = 1$.

## Credit

This codebase uses scripts from the [`salesforce/online_conformal`](https://github.com/salesforce/online_conformal) repository for the purpose of benchmarking existing algorithms and obtaining results.
