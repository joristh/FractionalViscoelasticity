
import torch

"""
==================================================================================================================
Mean Square Error objective function
==================================================================================================================
"""

class MSE:

    def __init__(self, data=0):
        self.data = data
        if not torch.is_tensor(self.data):
            self.data = torch.tensor(self.data)
        if len(self.data.shape)==1:
            self.data = self.data.reshape([-1,1])
            self.two_kernels = False
        else:
            self.two_kernels = True

    def __call__(self, y):
        if not torch.is_tensor(y):
            y = torch.tensor(y)
        # weights = torch.tesnor([1, 10]).double()
        # J = torch.sum( (y - self.data).square().sum(dim=0) / self.data.square().sum(dim=0) )
        J = (y - self.data).square().sum(dim=0) / self.data.square().sum(dim=0)
        if self.two_kernels:
            J = J[...,0] + 10*J[...,1]
        return 0.5*J
