
import numpy as np
import torch
from torch import nn
from math import gamma



"""
==================================================================================================================
Abstract kernel class (parent)
==================================================================================================================
"""

class AbstractKernel(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        parameters = kwargs.get("parameters", self.default_parameters(**kwargs)) ### list of the model parameters
        self._kernel_parameters = nn.Parameter(torch.zeros([len(parameters)], dtype=torch.float64))
        self.update_parameters(parameters)

    def default_parameters(self, **kwargs):
        return []

    def update_parameters(self, parameters=None):
        if parameters is not None:
            self._kernel_parameters.data[:] = parameters

    @np.vectorize
    def __call__(self, t):
        return 0

    @np.vectorize
    def eval_spectrum(self, z):  
        return 0


"""
==================================================================================================================
Fractional kernel class
==================================================================================================================
"""

class FractionalKernel(AbstractKernel):

    def __init__(self, **kwargs):
        super().__init__()
        alpha = kwargs.get("alpha", 0.5)
        self.update_parameters(parameters=[alpha])

    def default_parameters(self, **kwargs):
        return [0.5]

    def update_parameters(self, parameters=None):
        if parameters is not None:
            self._kernel_parameters.data[:] = torch.tensor(parameters).sqrt()
        self.alpha = self._kernel_parameters[0].square()

    @np.vectorize
    def __call__(self, t):
        return t**(self.alpha-1) / gamma(self.alpha)

    @np.vectorize
    def eval_spectrum(self, z):  
        return z**(-self.alpha)


        
"""
==================================================================================================================
Sum-of-exponentials kernel class
==================================================================================================================
"""

class SumOfExponentialsKernel(AbstractKernel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eval_func     = np.vectorize(self._eval_func)
        self.eval_spectrum = np.vectorize(self._eval_spectrum)


    ### Initialize parameters from a rational approximation
    def default_parameters(self, **kwargs):
        from .RationalApproximation import RationalApproximation_AAA as RationalApproximation
        settings = kwargs.get("init_fractional", {"alpha" : 0.5, "tol" : 1.e-4 }) ### dictionary of RA settings
        RA = RationalApproximation(**settings)
        parameters = list(RA.c) + list(RA.d)
        if kwargs.get("infmode", False):
            parameters.append(RA.c_inf)
        return parameters


    def update_parameters(self, parameters=None):
        if parameters is not None:
            self._kernel_parameters.data[:] = torch.tensor(parameters).double().sqrt() ### impose positivity via sqrt/square
            nparameters = self._kernel_parameters.data.numel()
            #assert( nparameters % 2 == 0 )
            self.nModes = nparameters // 2
            if (nparameters % 2 == 0):
                self.infmode_bool = False
            else:
                self.infmode_bool = True

        self.weights   = self._kernel_parameters[:self.nModes].square()
        if self.infmode_bool:
            self.exponents = self._kernel_parameters[self.nModes:-1].square()
            self.infmode   = self._kernel_parameters[-1].square()
        else:
            self.exponents = self._kernel_parameters[self.nModes:].square()
            self.infmode   = 0


    """
    ==================================================================================================================
    Functions evaluation
    ==================================================================================================================
    """

    # @np.vectorize
    def _eval_func(self, t):
        c, d = self.weights.detach().numpy(), self.exponents.detach().numpy()
        return np.sum(c * np.exp(-d*t))


    # @np.vectorize
    def _eval_spectrum(self, z):
        c, d, c_inf = self.weights.detach().numpy(), self.exponents.detach().numpy(), self.exponents.detach().numpy()
        return np.sum(c / (z + d)) + c_inf


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
        self.coef_a  = ( self.wk * self.coef_ak ).sum() + 2/h * self.infmode
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
