from collections import defaultdict
import numpy as np
import pandas as pd
from typing import Tuple

from online_conformal.base import BasePredictor
from online_conformal.enbpi import EnbMixIn
from online_conformal.utils import pinball_loss_grad

from scipy.special import erfi

# variables to update: 
# gradient variance: v
# gradient sum: s
# running estimate of the Lipshitz constant: h
# prediction: x

class MagLearnUndiscounted(BasePredictor):
    """
    Undiscounted 1D Magnitude Learner
    """

    def __init__(self, *args, horizon=1, max_scale=None, **kwargs):
        self.scale = {}
        self.delta = defaultdict(float)
        # self.grad_norm = defaultdict(float)
        self.grad = 0.0 # gradient
        
        self.v = 0 # gradient variance
        self.s = 0 # gradient sum
        self.h = 0 # running estimate of Lipshitz constant
        self.delta_unproj = 0 # unprojected prediction

        super().__init__(*args, horizon=horizon, **kwargs) # initialize base predictor

        
    def erfi_unscaled(self,z):
        return erfi(z)/(2/np.sqrt(np.pi)) # unscaled erfi function
    
    def predict(self,horizon) -> Tuple[float, float]: 
        return -self.delta[horizon], self.delta[horizon] # return the radius of the prediction interval
    
    def update(self, ground_truth: pd.Series, forecast: pd.Series, horizon): # update the prediction radius
        residuals = np.abs(ground_truth - forecast).values # difference between ground truth and forecast
        self.residuals.extend(horizon, residuals.tolist()) # add residuals to the list of residuals
        EPSILON = 1 # hyperparameter
        for s in residuals: # for each residual
            delta = self.delta[horizon] # get the prediction radius
            if self.h == 0: # if the estimated Lipshitz constant is zero
                self.delta_unproj = 0 # unprojected
            else: # if the estimated Lipshitz constant is not zero, compute the unprojected prediction
                self.delta_unproj = EPSILON * self.erfi_unscaled(self.s/(2*np.sqrt(self.v + 2*self.h * self.s + 16*self.h**2))) - \
                EPSILON * (self.h / np.sqrt(self.v + 2*self.h * self.s + 16 * self.h**2) )* np.exp(self.s**2/(4*(self.v + 2*self.h*self.s + 16*self.h**2)))
            
            delta = np.clip(self.delta_unproj, 0, np.inf) # clip the unprojected prediction to be non-negative
            
            grad = pinball_loss_grad(np.abs(s), delta, self.coverage) # get the gradient
            
            if grad*self.delta_unproj < grad*delta:
                # in practice, this condition shouldn't be entered.
                grad_surrogate = 0 
            else: # this condition should be entered every time
                grad_surrogate = grad

            grad_surr_clipped = np.clip(grad_surrogate, -self.h, self.h) # clip the gradient to be within the range of the Lipshitz constant
            self.h = max(self.h, np.abs(grad_surrogate)) # update the running estimate of the Lipshitz constant
            self.v = self.v + (grad_surr_clipped)**2 # update the gradient variance
            self.s = self.s - grad_surr_clipped # update the gradient sum
            self.delta[horizon] = delta # update the prediction radius
    
class EnbMagLearnUndiscounted(EnbMixIn, MagLearnUndiscounted):
    pass