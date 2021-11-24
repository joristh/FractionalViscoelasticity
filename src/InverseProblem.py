




"""
==================================================================================================================
Inverse problem using torch optimizer
==================================================================================================================
"""


import torch
from torch.nn.utils.convert_parameters import parameters_to_vector
from ufl.measure import register_integral_type


class InverseProblem:

    def __init__(self, *args, **kwargs):
        # self.calibrate(*args, **kwargs)
        pass

    def __call__(self, *args, **kwargs):
        self.calibrate(*args, **kwargs)


    def calibrate(self, Model, objective, initial_guess=None, **kwargs):
        verbose  = kwargs.get("verbose", False)
        max_iter = kwargs.get("max_iter", 20)
        lr       = kwargs.get("lr",  1.)
        tol      = kwargs.get("tol", 1.e-3)
        reg      = kwargs.get("regularization", None)

        Model.fg_inverse = True
        Model.fg_export  = False

        if initial_guess is not None:
            for i, p in enumerate(Model.parameters()):
                p.data[:] = torch.tensor(initial_guess[i])

        ### print initial parameters
        self.print_parameters(Model.parameters())

        ### Optimizer
        optimizer = kwargs.get("optimizer", torch.optim.SGD)
        if optimizer is torch.optim.SGD:
            if verbose:
                print()
                print('=================================')
                print('        Gradient descent         ')
                print('=================================')
                print()
            nepochs = max_iter
            optimization_settings = {
                'lr'    :   lr,
            }
        elif optimizer is torch.optim.LBFGS:
            if verbose:
                print()
                print('=================================')
                print('             LBFGS               ')
                print('=================================')
                print()
            nepochs = kwargs.get("nepochs", 1)
            optimization_settings = {
                'lr'                :   lr,
                'line_search_fn'    :   'strong_wolfe',
                'max_iter'          :   max_iter,
            }       
        self.Optimizer = optimizer(Model.parameters(), **optimization_settings)

        ### Convergence history
        self.convergence_history = {
            'loss'      :   [],
            'grad'      :   [],
            'loss_all'  :   [],  ### values including internal BFGS and line search iterations
        }


        def closure():
            self.Optimizer.zero_grad()
            theta = parameters_to_vector(Model.parameters())
            Model.forward_solve()
            obs = Model.observations
            self.loss = objective(obs)
            if reg: self.loss = self.loss + reg(theta)
            self.loss.backward()
            print('loss = ', self.loss.item())
            self.convergence_history['loss_all'].append(self.loss.item())
            return self.loss

        def get_grad():
            return torch.cat([p.grad for p in Model.parameters()])


        ### Optimization loop
        for epoch in range(nepochs):
            self.Optimizer.step(closure)
            g_norm = get_grad().norm(p=float('inf'))

            ### convergence monitor
            if verbose:
                print()
                print('=================================')
                print('-> Epoch {0:d}/{1:d}'.format(epoch+1, nepochs))
                print('=================================')
                print('loss = ', self.loss.item())
                print('grad = ', g_norm.item())
                print('=================================')
                print()
                print()

            ### store history
            self.convergence_history['loss'].append(self.loss.item())
            self.convergence_history['grad'].append(g_norm.item())

            ### stopping criterion
            if g_norm < tol: break

        ### ending
        theta_opt = parameters_to_vector(Model.parameters())
        Model.fg_inverse = False
        return theta_opt



    def print_parameters(self, parameters):
        # print("Parameters: ", [p.tolist() for p in parameters])
        weights, exponents = [p for p in parameters]
        print("Weights:   ", weights.tolist())
        print("Exponents: ", exponents.tolist())
            



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