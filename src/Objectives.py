
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

    def __call__(self, y):
        J = 0.5*(y - self.data).square().sum()
        return J
