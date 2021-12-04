




"""
==================================================================================================================
Inverse problem using torch optimizer
==================================================================================================================
"""


import torch
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
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
        self.fg_split_kernels = Model.fg_split_kernels

        if initial_guess is not None:
            for i, p in enumerate(Model.parameters()):
                p.data[:] = torch.tensor(initial_guess[i]).sqrt()

        ### print initial parameters
        print('Number of parameters =', sum(p.numel() for p in Model.parameters()))
        print('Initial parameters:')
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
                'line_search_fn'    :   kwargs.get('line_search_fn', 'strong_wolfe'),
                'max_iter'          :   max_iter,
                'tolerance_grad'    :   tol,
                # 'tolerance_change'  :   tol,
                'history_size'      :   kwargs.get('history_size', 100),
            }       
        self.Optimizer = optimizer(Model.parameters(), **optimization_settings)

        ### Convergence history
        self.convergence_history = {
            'loss'      :   [],
            'grad'      :   [],
            'parameters':   [],
        }


        ### Loading type
        loading = kwargs.get("loading", None)
        if isinstance(loading, list): ### multiple loadings case
            def Forward():
                obs = torch.tensor([])
                for loading_instance in loading:
                    Model.forward_solve(loading=loading_instance)
                    obs = torch.cat([obs, Model.observations])
                return obs
        else:
            def Forward():
                Model.forward_solve()
                obs = Model.observations
                return obs


        def get_grad():
            return torch.cat([p.grad for p in Model.parameters()])

        self.iter = 0
        def closure():
            self.Optimizer.zero_grad()
            theta     = parameters_to_vector(Model.parameters())
            obs       = Forward()
            self.loss = objective(obs)
            if reg: ### regularization term
                self.loss = self.loss + reg(theta)
            self.loss.backward()
            self.grad = get_grad()
            grad_norm = self.grad.norm(p=float('inf'))

            ### convergence monitor
            self.iter = self.iter + 1
            if verbose:
                print()
                print('=================================')
                print('-> Iteration {0:d}/{1:d}'.format(self.iter, max_iter))
                print('=================================')
                print('loss = ', self.loss.item())
                print('grad = ', grad_norm.item())
                print('=================================')
                print('parameters:')                
                self.print_parameters(Model.parameters())
                print('=================================')
                print()
                print()

            ### store convergence history
            self.convergence_history['loss'].append(self.loss.item())
            self.convergence_history['grad'].append(grad_norm.item())
            self.convergence_history['parameters'].append(parameters_to_vector(Model.parameters()))

            return self.loss


        ### Minimization
        for epoch in range(nepochs):
            self.Optimizer.step(closure)


        ### ending
        theta_opt = parameters_to_vector(Model.parameters()).square()
        Model.fg_inverse = False
        return theta_opt



    def print_parameters(self, parameters):
        # print("Parameters: ", [p.tolist() for p in parameters])
        if self.fg_split_kernels:
            p = parameters_to_vector(parameters).square()
            n = len(p) // 4
            weights, exponents = p[:n], p[n:2*n]
            print("Ker 1: Weights:   ", weights.tolist())
            print("Ker 1: Exponents: ", exponents.tolist())
            weights, exponents = p[2*n:3*n], p[3*n:]
            print("Ker 2: Weights:   ", weights.tolist())
            print("Ker 2: Exponents: ", exponents.tolist())
        else:
            weights, exponents = [p.square() for p in parameters]
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