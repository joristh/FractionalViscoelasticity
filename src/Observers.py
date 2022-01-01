from fenics import *
from fenics_adjoint import *
import torch_fenics


"""
==================================================================================================================
Abstract observer
==================================================================================================================
"""

class AbstractObserver(torch_fenics.FEniCSModule):

    def __init__(self, **kwargs):
        super().__init__()

    def observation(self, Model):
        pass

    def solve(self, *inputs):
        
        ### TODO: your code
        output = None

        return output

    def input_templates(self):
        ### TODO: your inputs template
        return Constant(0.)



"""
==================================================================================================================
Tip displacement observer
==================================================================================================================
"""

class TipDisplacementObserver(AbstractObserver):

    def __init__(self, Model):
        super().__init__()
        ds = Model.ds_Neumann
        self.surface_measure = ds
        self.tip_area = assemble(1.*ds)
        self.Model = Model

    def observe(self):
        u = self.Model.u.reshape([1, -1, self.Model.ndim])
        u_tip = self.__call__(u)
        return u_tip

    def solve(self, u):
        ds, S = self.surface_measure, self.tip_area
        # u_comp = split(u)
        # u_tip  = assemble(u_comp[1]*ds) / S
        u_norm = sqrt(dot(u, u))
        u_tip  = assemble(u_norm*ds) / S
        return u_tip

    def input_templates(self):
        return Function(self.Model.V)
