




"""
==================================================================================================================
Inverse problem using torch optimizer
==================================================================================================================
"""


import torch
from torch.nn.utils.convert_parameters import parameters_to_vector


class InverseProblem:

    def __init__(self, **kwargs) -> None:
        self.objective = kwargs.get("objective", None)
        self.Reg       = kwargs.get("regularization", None)  

    def __call__(self, Model):
        self.calibrate(Model)


    def calibrate(self, Model, data=None, initial_guess=None, **kwargs):
        verbose = kwargs.get("verbose", False)
        lr      = kwargs.get("lr",  1.)
        tol     = kwargs.get("tol", 1.e-3)

        Model.fg_export = False

        if initial_guess is not None:
            for i, p in enumerate(Model.parameters()):
                p.data[:] = initial_guess[i]
            

        self.Optimizer = kwargs.get('Optimizer', torch.optim.sgd)
        self.Optimizer = self.Optimizer(Model.parameters(), lr=lr) #, line_search_fn='strong_wolfe')

        def closure():
            self.Optimizer.zero_grad()
            theta = parameters_to_vector(Model.parameters())
            self.loss = self.objective(theta)
            if self.Reg:
                self.loss = self.loss + self.Reg(theta)
            self.loss.backward()
            if verbose:
                print('loss = ', self.loss.item())
                # self.print_grad()
                # self.print_parameters()
                # self.LossFunc.Model.plot()
            return self.loss

        nepochs = 10
        for epoch in range(nepochs):
            if verbose:
                print()
                print('=================================')
                print('-> Epoch {0:d}'.format(epoch))
                print('=================================')
            self.Optimizer.step(closure)
            # if self.verbose:
            #     self.print_grad()
            #     self.print_parameters()
            #     print()
            if self.loss.item() < tol: break


        theta_opt = parameters_to_vector(Model.parameters())

        # Model.ViscousTerm.update_parameters(theta_opt)

        # self.optimal_parameters = theta_opt
        # self.final_objective    = self.objective(Model, theta, data)
        # self.Model = Model

        return theta_opt



"""
==================================================================================================================
Inverse problem using pyadjoint optimizer
==================================================================================================================
"""


from fenics import *
from fenics_adjoint import *


class InverseProblem_pyadjoint:

    def __init__(self, **kwargs) -> None:
        self.objective = kwargs.get("objective", None)        

    def __call__(self, Model):
        self.calibrate(Model)


    def calibrate(self, Model, data=None, initial_guess=None):

        Model.fg_export = False

        if initial_guess is None:
            initial_guess = Model.ViscousTerm.parameters()

        theta = [ AdjFloat(theta_k) for theta_k in  initial_guess]
        # theta = ndarray(2*Model.ViscousTerm.nModes)
        # theta[:] = Model.ViscousTerm.parameters()

        J = self.objective(Model, theta, data)

        
        control = [ Control(theta_k) for theta_k in theta ]
        # control = Control(theta)
        Jhat  = ReducedFunctional(J, control)
        Jhat(theta)

        # tape = get_working_tape()
        # tape.visualise()

        theta_opt = minimize(Jhat, options={"disp": True})

        # Model.ViscousTerm.update_parameters(theta_opt)

        # self.optimal_parameters = theta_opt
        # self.final_objective    = self.objective(Model, theta, data)
        # self.Model = Model

        return theta_opt