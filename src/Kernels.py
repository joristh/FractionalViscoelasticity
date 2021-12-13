import torch
from torch import nn
import numpy as np


"""
==================================================================================================================
Abstract kernel class (parent)
==================================================================================================================
"""

class AbstractKernel:

    def __init__(self) -> None:
        pass


    def __call__(self, t):
        return 0
        


"""
==================================================================================================================
Sum-of-exponentials kernel class
==================================================================================================================
"""

class SumOfExponentialsKernel(AbstractKernel):

    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.nModes    = kwargs.get("nModes", 1)
        self.Weights   = nn.Parameter(torch.ones([self.nModes], dtype=torch.float64))
        self.Exponents = nn.Parameter(torch.zeros([self.nModes], dtype=torch.float64))

        weights = kwargs.get("weights", None)
        if weights is not None: self.set_Weights(weights)

        exponents = kwargs.get("exponents", None)
        if exponents is not None: self.set_Exponents(exponents)

    
    def __call__(self, t):       
        return torch.sum( self.Weights * torch.exp(-t*self.Exponents) )

    def eval_spectrum(self, z):       
        return torch.sum( self.Weights / (z + self.Exponents) )


    def set_Weights(self, values):
        for k in range(self.nModes):
            self.Weights.data[k] = values[k]

    def set_Exponents(self, values):
        for k in range(self.nModes):
            self.Exponents.data[k] = values[k]  

    
    def parameters(self):
        p = [0.]*(2*self.nModes)
        for k in range(self.nModes):
            p[k]               = self.Weights.data[k]
            p[self.nModes + k] = self.Exponents.data[k]
        return p


    def update_parameters(self, parameters):
        weights   = parameters[:self.nModes]
        exponents = parameters[self.nModes:]
        self.set_Weights(weights)
        self.set_Exponents(exponents)
        self.compute_coefficients(self.h)


    def compute_coefficients(self, h, gamma=1):
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
        self.h = h


    def update_history(self, VecFn1):
        Fn1 = torch.tensor(VecFn1).unsqueeze(dim=-1)

        if not hasattr(self, "modes"):
            self.modes = torch.zeros([Fn1.shape[0], self.nModes])
            self.Fn    = torch.zeros_like(Fn1)

        h = self.h

        self.modes   = self.coef_bk * self.modes + 0.5*h*self.coef_ck*self.Fn + 0.5*h*self.coef_ak*Fn1
        self.history = ( self.wk * self.coef_bk * self.modes ).sum(dim=-1)
        self.Fn[:]   = Fn1

        return self.history #.detach().numpy()


"""
==================================================================================================================
Sum-of-exponentials kernel class (torch version)
==================================================================================================================
"""

class SumOfExponentialsKernel_Torch(nn.Module):

    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.nModes    = kwargs.get("nModes", 1)
        self.Weights   = nn.Parameter(torch.ones([self.nModes], dtype=torch.float64))
        self.Exponents = nn.Parameter(torch.zeros([self.nModes], dtype=torch.float64))
        self.InfMode   = nn.Parameter(torch.special.logit(torch.zeros(1, dtype=torch.float64)))

        weights = kwargs.get("weights", None)
        if weights is not None: self.set_Weights(weights)

        exponents = kwargs.get("exponents", None)
        if exponents is not None: self.set_Exponents(exponents)

        infmode = kwargs.get("infmode", None)
        if infmode is not None: self.set_InfMode(infmode)
        else: self.InfMode.requires_grad_(False)
        
    
    def __call__(self, t):       
        return torch.sum( self.Weights * torch.exp(-t*self.Exponents) )

    def eval_spectrum(self, z):       
        return torch.sum( self.Weights / (z + self.Exponents) )


    def set_Weights(self, values):
        for k in range(self.nModes):
            self.Weights.data[k] = np.sqrt(values[k])

    def set_Exponents(self, values):
        for k in range(self.nModes):
            self.Exponents.data[k] = np.sqrt(values[k])

    def set_InfMode(self, value):
        self.InfMode.data[0] = torch.special.logit(torch.tensor(value))


    def update_parameters(self, parameters):
        weights   = parameters[:self.nModes]
        exponents = parameters[self.nModes:-1]
        inf_mode  = parameters[-1]
        self.set_Weights(weights)
        self.set_Exponents(exponents)
        self.set_InfMode(inf_mode)
        self.compute_coefficients(self.h)


    def compute_coefficients(self, h=None, gamma=1):
        if h is None:
            h = self.h
        else:
            self.h = h
        lmbda   = self.Exponents.square()
        theta   = lmbda / (1 + lmbda)
        self.wk = self.Weights.square() * (1-theta)
        lgh     = lmbda*gamma*h
        den     = (1-theta)*(1 + lgh) + theta * h/2 * (1 + 2*lgh)
        self.coef_ak = (1 + 2*lgh) / den
        self.coef_bk = ( (1-theta)*(1+lgh) - theta * h/2 ) / den
        self.coef_ck = 1 / den
        self.coef_a  = ( self.wk * self.coef_ak ).sum() + 2/h*torch.special.expit(self.InfMode)
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
