
import numpy as np
import torch
from torch import nn



"""
==================================================================================================================
Abstract kernel class (parent)
==================================================================================================================
"""

class AbstractKernel(nn.Module):

    def __init__(self, **kwargs) -> None:
        super().__init__()
        parameters       = kwargs.get("parameters", self.default_parameters(**kwargs)) ### list of the model parameters
        self._parameters = nn.Parameter(torch.zeros([len(parameters)], dtype=torch.float64))
        self.update_parameters(parameters)

    def default_parameters(self, **kwargs):
        return []

    def update_parameters(self, parameters=None):
        if parameters is not None:
            self._parameters.data[:] = parameters

    @np.vectorize
    def __call__(self, t):
        return 0

    @np.vectorize
    def eval_spectrum(self, z):  
        return 0
        
"""
==================================================================================================================
Sum-of-exponentials kernel class
==================================================================================================================
"""

class SumOfExponentialsKernel(AbstractKernel):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.nModes = self.weights.numel()


    ### Initialize parameters from a rational approximation
    def default_parameters(self, **kwargs):
        from .RationalApproximation import RationalApproximation_AAA as RationalApproximation
        settings = kwargs.get("init_fractional", {"alpha" : 0.5, "tol" : 1.e-4 }) ### dictionary of RA settings
        RA = RationalApproximation(**settings)
        parameters = list(RA.c) + list(RA.d)
        return parameters


    def update_parameters(self, parameters=None):
        if parameters is not None:
            self._parameters.data[:] = parameters
            self._parameters.data[:] = self._parameters.data.sqrt() ### impose positivity via sqrt/square
        self.weights   = self._parameters.data[:self.nModes].square()
        self.exponents = self._parameters.data[self.nModes:].square()
        self.compute_coefficients(self.h)


    """
    ==================================================================================================================
    Functions evaluation
    ==================================================================================================================
    """

    @np.vectorize
    def __call__(self, t):
        c, d = self.weights.detach().numpy(), self.exponents.detach().numpy()
        return np.sum(c * np.exp(-d*t))


    @np.vectorize
    def eval_spectrum(self, z):
        c, d = self.weights.detach().numpy(), self.exponents.detach().numpy()     
        return np.sum(c / (z + d))


    """
    ==================================================================================================================
    Evolution of the modes and the history integral
    ==================================================================================================================
    """

    def compute_coefficients(self, h=None, gamma=1):
        if h is None:
            h = self.h
        else:
            self.h = h
        lmbda   = self.exponents
        theta   = lmbda / (1 + lmbda)
        self.wk = self.weights * (1-theta)
        lgh     = lmbda*gamma*h
        den     = (1-theta)*(1 + lgh) + theta * h/2 * (1 + 2*lgh)
        self.coef_ak = (1 + 2*lgh) / den
        self.coef_bk = ( (1-theta)*(1+lgh) - theta * h/2 ) / den
        self.coef_ck = 1 / den
        self.coef_a  = ( self.wk * self.coef_ak ).sum()
        self.coef_c  = ( self.wk * self.coef_ck ).sum()


    def init(self, h=None, gamma=1):
        self.update_parameters()
        self.compute_coefficients(h, gamma)
        self.modes   = None
        self.history = 0
        


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
