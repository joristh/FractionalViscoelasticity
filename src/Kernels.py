
import numpy as np
import torch
from torch import nn



"""
==================================================================================================================
Abstract kernel class (parent)
==================================================================================================================
"""

class AbstractKernel(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()      


    def __call__(self, t):
        return 0
        
"""
==================================================================================================================
Sum-of-exponentials kernel class
==================================================================================================================
"""

class SumOfExponentialsKernel(AbstractKernel):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        
        parameters       = kwargs.get("parameters", self.init_fractional(**kwargs)) ### list of the model parameters (default: from fractional)
        self._parameters = nn.Parameter(torch.tensor(parameters, dtype=torch.float64).sqrt())  ### impose positivity via sqrt/square
        self.nModes      = len(parameters)
        self.Weights   = nn.Parameter(torch.ones([self.nModes], dtype=torch.float64))
        self.Exponents = nn.Parameter(torch.zeros([self.nModes], dtype=torch.float64))

        weights = kwargs.get("weights", None)
        if weights is not None: self.set_Weights(weights)

        exponents = kwargs.get("exponents", None)
        if exponents is not None: self.set_Exponents(exponents)


    ### Initialize parameters from a rational approximation
    def init_fractional(self, **kwargs):
        from .RationalApproximation import RationalApproximation_AAA as RationalApproximation
        settings = kwargs.get("init_fractional", {"alpha" : 0.5 }) ### dictionary of RA settings
        RA = RationalApproximation(**settings)
        parameters = list(RA.c) + list(RA.d)
        return parameters


    def set_Weights(self, values):
        for k in range(self.nModes):
            self.Weights.data[k] = values[k]

    def set_Exponents(self, values):
        for k in range(self.nModes):
            self.Exponents.data[k] = values[k]


    def update_parameters(self, parameters):
        weights   = parameters[:self.nModes]
        exponents = parameters[self.nModes:]
        self.set_Weights(weights)
        self.set_Exponents(exponents)
        self.compute_coefficients(self.h)

    

    @np.vectorize
    def __call__(self, t):
        c, d = self.Weights.detach().numpy(), self.Exponents.detach().numpy()
        return np.sum(c * np.exp(-d*t))


    @np.vectorize
    def eval_spectrum(self, z):
        c, d = self.Weights.detach().numpy(), self.Exponents.detach().numpy()     
        return np.sum(c / (z + d))



    def compute_coefficients(self, h=None, gamma=1):
        if h is None:
            h = self.h
        else:
            self.h = h
        lmbda   = self.Exponents
        theta   = lmbda / (1 + lmbda)
        self.wk = self.Weights * (1-theta)
        lgh     = lmbda*gamma*h
        den     = (1-theta)*(1 + lgh) + theta * h/2 * (1 + 2*lgh)
        self.coef_ak = (1 + 2*lgh) / den
        self.coef_bk = ( (1-theta)*(1+lgh) - theta * h/2 ) / den
        self.coef_ck = 1 / den
        self.coef_a  = ( self.wk * self.coef_ak ).sum()
        self.coef_c  = ( self.wk * self.coef_ck ).sum()


    def init(self, h=None, gamma=1):
        self.compute_coefficients(h, gamma)
        self.modes = None
        


    def update_history(self, F):
        F_new = F.reshape([-1, 1])

        if (not hasattr(self, "modes")) or (self.modes is None):
            self.modes = torch.zeros([F_new.shape[0], self.nModes])
            self.F_old = torch.zeros_like(F_new)

        h = self.h

        self.modes   = self.coef_bk * self.modes + 0.5*h*self.coef_ck*self.F_old + 0.5*h*self.coef_ak*F_new
        self.history = ( self.wk * self.coef_bk * self.modes ).sum(dim=-1)
        self.F_old   = 1.*F_new

        return self.history
