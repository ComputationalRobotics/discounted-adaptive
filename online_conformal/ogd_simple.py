#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
#
from collections import defaultdict
import numpy as np
import pandas as pd
from typing import Tuple

from online_conformal.base import BasePredictor
from online_conformal.utils import pinball_loss_grad


class SimpleOGD(BasePredictor):
    """
    Scale-free online gradient descent to learn conformal confidence intervals via online quantile regression. We
    perform online gradient descent on the pinball loss to learn the relevant quantiles of the residuals.
    From Orabona & Pal, 2016, "Scale-Free Online Learning." https://arxiv.org/abs/1601.01974.
    """

    def __init__(self, *args, horizon=1, max_scale=None, **kwargs):
        self.scale = {} # learning rate
        self.delta = defaultdict(float) # prediction radius
        self.grad_norm = defaultdict(float) # normalized gradient
        super().__init__(*args, horizon=horizon, **kwargs) # initialize base predictor
        self.scale[0] = 1 # learning rate
        self.scale[1] = 1 # learning rate

    def predict(self, horizon) -> Tuple[float, float]:
        return -self.delta[horizon], self.delta[horizon] # return the radius of the prediction interval

    def update(self, ground_truth: pd.Series, forecast: pd.Series, horizon):
        residuals = np.abs(ground_truth - forecast).values # difference between ground truth and forecast
        self.residuals.extend(horizon, residuals.tolist()) # add residuals to the list of residuals
        if horizon not in self.scale: # if horizon is not in the scale
            return
        for s in residuals: # for each residual
            delta = self.delta[horizon] # get the prediction radius
            grad = pinball_loss_grad(np.abs(s), delta, self.coverage) # get the gradient
            self.grad_norm[horizon] += grad**2 # update the gradient norm
            if self.grad_norm[horizon] != 0: # if the gradient norm is not zero
                self.delta[horizon] = max(0, delta - self.scale[horizon] / np.sqrt(3 * self.grad_norm[horizon]) * grad) # update the prediction radius