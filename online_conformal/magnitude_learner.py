from collections import defaultdict
import numpy as np
import pandas as pd
from typing import Tuple

from online_conformal.base import BasePredictor
from online_conformal.enbpi import EnbMixIn
from online_conformal.utils import pinball_loss_grad, Residuals

from scipy.special import erfi

# variables to update: 
# gradient variance: v
# gradient sum: s
# running estimate of the Lipshitz constant: h
# prediction: x

class MagnitudeLearner(BasePredictor):
    """
    1D MagnitudeLearner
    """

    def __init__(self, *args, horizon=1, max_scale=None, **kwargs):
        self.scale = {}
        self.delta = defaultdict(float)
        self.grad_norm = defaultdict(float)
        self.grad = 0.0
        
        self.v = 0 # gradient variance
        self.s = 0 # gradient sum
        self.h = 0 # running estimate of Lipshitz constant
        self.delta_unproj = 0 # unprojected prediction
        
        # if max_scale is None:
        #     self.scale = {}
        # else:
        #     self.scale = {j + 1: float(max_scale) for j in range(horizon)}
        super().__init__(*args, horizon=horizon, **kwargs)
        
    def erfi_unscaled(self,z):
        return erfi(z)/(2/np.sqrt(np.pi))
    
    def predict(self,horizon) -> Tuple[float, float]:
        return -self.delta[horizon], self.delta[horizon]
    
    def update(self, ground_truth: pd.Series, forecast: pd.Series, horizon):
        residuals = np.abs(ground_truth - forecast).values
        self.residuals.extend(horizon, residuals.tolist())
        # if horizon not in self.scale:
        #     return
        #EPSILON = 10000
        #EPSILON = 10
        EPSILON = 10
        DISCOUNT_FACTOR = 0.999
        for s in residuals:
            #print("s = ", s)
            delta = self.delta[horizon]
            # Get the unprojected prediction x_tilde
            if self.h == 0:
                self.delta_unproj = 0 # unprojected
            else: 
                self.delta_unproj = EPSILON * self.erfi_unscaled(self.s/(2*np.sqrt(self.v + 2*self.h * self.s + 16*self.h**2))) - \
                EPSILON * (self.h / np.sqrt(self.v + 2*self.h * self.s + 16 * self.h**2) )* np.exp(self.s**2/(4*(self.v + 2*self.h*self.s + 16*self.h**2)))
            delta = np.clip(self.delta_unproj, 0, np.inf)
            
            grad = pinball_loss_grad(np.abs(s), delta, self.coverage)
            
            # if grad*self.delta_unproj < grad*delta:
            #     # in practice, this condition shouldn't be entered.
            #     grad_surrogate = grad
            #     #grad_surrogate = 0
            #     print("This condition should be not be entered.")
            #     #assert(False)
            # else:
            #     grad_surrogate = grad
            grad_surrogate = grad

            grad_surr_clipped = np.clip(grad_surrogate, -DISCOUNT_FACTOR*self.h, DISCOUNT_FACTOR*self.h)
            self.h = max(DISCOUNT_FACTOR*self.h, np.abs(grad_surrogate))
            self.v = (DISCOUNT_FACTOR**2) * self.v + (grad_surr_clipped)**2
            self.s = DISCOUNT_FACTOR * self.s - grad_surr_clipped
            self.delta[horizon] = delta
            # print("delta_unproj = ", self.delta_unproj)
            # print("delta = ", self.delta[horizon])
            # print("grad = ", grad)
            # print("grad_surrogate = ", grad_surrogate)
    
class EnbMagnitudeLearner(EnbMixIn, MagnitudeLearner):
    pass