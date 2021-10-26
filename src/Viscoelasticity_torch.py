
from fenics import *
from fenics_adjoint import *
# from numpy_adjoint import *
import torch_fenics
import fenics_adjoint

import numpy as np

import torch
from torch import nn

from tqdm import tqdm
import matplotlib.pyplot as plt

from .Kernels import SumOfExponentialsKernel_Torch
# from .reduced_function import ReducedFunctionTorch


"""
==================================================================================================================
Elastoplasticity forward problem class
==================================================================================================================
"""

class ViscoelasticityProblem(torch_fenics.FEniCSModule):

    def __init__(self, **kwargs):
        super().__init__()

        self.verbose      = kwargs.get("verbose", False)
        self.fg_export    = kwargs.get("export", None)
        self.inputfolder  = kwargs.get("inputfolder", "./")
        self.outputfolder = kwargs.get("outputfolder", "./")
        self.fg_viscosity = kwargs.get("viscosity", False)
        self.fg_inverse   = kwargs.get("InverseProblem", False)

        # elastic parameters
        E  = kwargs.get('Young', 1.e3)
        nu = kwargs.get('Poisson', 0.3)
        E, nu = Constant(E), Constant(nu)
        lmbda = E*nu/(1+nu)/(1-2*nu)
        mu    = E/2./(1+nu)
        self.lmbda, self.mu = lmbda, mu

        # Mass density
        rho = kwargs.get('density', 1.)
        rho = Constant(rho)
        self.rho = rho


        ### Mesh
        mesh = self.set_mesh(**kwargs)



        deg_u = kwargs.get("degree", 1)
        V = VectorFunctionSpace(mesh, "CG", deg_u)
        self.V = V

        # u  = Function(V, name="displacement")
        # v  = Function(V, name="velocity")
        # a  = Function(V, name="acceleration")
        # a_new = Function(V, name="new acceleration")
        # inc= Function(V, name="Newton increment")
        # w  = Function(V, name="axilary variable")
        # w_tr  = Function(V, name="axilary variable (trace)")
        # w_dev = Function(V, name="axilary variable (deviator)")
        
        # self.u, self.v, self.a = u, v, a
        # self.a_new = a_new
        # self.w = w


        self.set_boundary_condition(**kwargs)


        ### Assembling the FE matrices
        u_, v_  = TrialFunction(V), TestFunction(V)
        # self.M_form = rho * inner(u_, v_)*dx
        # self.K_form = inner(self.sigma(self.eps(u_)), self.eps(v_))*dx
        # self.M = assemble(self.M_form, annotate=True)
        # self.K = assemble(self.K_form, annotate=True)

        # self.fg_viscosity = kwargs.get("viscosity", False)
        # self.K_visc = self.K


        ### Source terms

        ### 1) body force
        body_force = kwargs.get("body_force", Constant((0.,)*self.ndim) )
        body_force_form = inner(body_force, v_)*dx
        self.f_vol = assemble(body_force_form)
        self.forces_form = self.f_vol

        ### 2) loading
        if self.NeumannBC:
            self.loading = kwargs.get("loading", Constant((0.,)*self.ndim) )
            self.loading_form  = inner(self.loading, v_)*self.ds_Neumann
            self.forces_form   = body_force_form + self.loading_form

        


        ### Time scheme
        self.Newmark = Newmark()
        self.set_time_stepper(**kwargs)


        ### Kernel
        self.kernel = SumOfExponentialsKernel_Torch(**kwargs)
        self.kernel.compute_coefficients(self.dt)


        ### History term
        # self.history = Vector(self.u.vector())
        # self.history[:] = self.kernel.update_history(self.v.vector().get_local()).detach().numpy()

        ### Linear solver
        self.LinSolver = set_linSolver()

        ### Observations and QoIs
        observer = kwargs.get("observer", None)
        if observer:
            self.observer = observer(Model=self)



        
        

    """
    ==================================================================================================================
    """

    def set_mesh(self, **kwargs):
        mesh = kwargs.get("mesh", None)
        if mesh is None: ### default mesh
            self.ndim = kwargs.get("ndim", 2)
            if self.ndim == 1:
                mesh = UnitIntervalMesh(20)
            elif self.ndim == 2:
                mesh = UnitSquareMesh(20,20)
            elif self.ndim == 3:
                mesh = UnitCubeMesh(20,20,20)
            else:
                raise Exception("Dimension {} is unsupported.".format(self.ndim))
        elif type(mesh) == str:
            mesh = Mesh(mesh)
        else:
            ### mesh = mesh
            pass
        self.mesh = mesh
        self.ndim = mesh.topology().dim()
        return mesh



    def set_boundary_condition(self, **kwargs):
        self.bc_a = None ### Homogeneous Neumann by default

        DirichletBoundary = kwargs.get("DirichletBoundary", None)
        NeumannBoundary   = kwargs.get("NeumannBoundary",   None)

        if NeumannBoundary:
            boundary_subdomains = MeshFunction("size_t", self.mesh, self.ndim - 1)
            boundary_subdomains.set_all(0)
            boundary_subdomain_Neumann = AutoSubDomain(NeumannBoundary)
            boundary_subdomain_Neumann.mark(boundary_subdomains, 1)
            # self.ds_Neumann = ds(subdomain_data=boundary_subdomains)#, metadata={"quadrature_degree": 3})(1) # Define measure for boundary condition integral
            self.ds_Neumann = Measure("ds", domain=self.mesh, subdomain_data=boundary_subdomains)(1)#, metadata={"quadrature_degree": 3})(1) # Define measure for boundary condition integral
            self.NeumannBC = True

        if DirichletBoundary:
            zero_value = Constant((0.,)*self.ndim)
            value      = kwargs.get("DirichletValue",  zero_value)
            self.bc_u  = DirichletBC(self.V, value,      DirichletBoundary)
            self.bc_a  = DirichletBC(self.V, zero_value, DirichletBoundary)
            self.DirichletBC = True




    def set_time_stepper(self, nTimeSteps=10, InitialTime=0, FinalTime=1, **kwargs):
        assert(InitialTime<FinalTime)
        self.time_steps = np.linspace(InitialTime, FinalTime, nTimeSteps+1)[1:]
        self.dt = (FinalTime-InitialTime) / nTimeSteps


    """
    ==================================================================================================================
    """

    def eps(self, v):
        e = sym(grad(v))
        return e

    def sigma(self, eps_el):
        return self.lmbda*tr(eps_el)*Identity(3) + 2*self.mu*eps_el

    def as_3D_tensor(self, X):
        return as_tensor([[X[0], X[3], 0],
                        [X[3], X[1], 0],
                        [0, 0, X[2]]])

    # Mass form
    def m(self, u, u_):
        return self.rho*inner(u, u_)*dx

    # Elastic stiffness form
    def k(self, u, u_):
        return inner(self.sigma(self.eps(u)), self.eps(u_))*dx

    # Viscous form 
    def c(self, u, u_):
        return self.k(u, u_)

    """
    ==================================================================================================================
    """

    def initialize_state(self):

        if self.fg_inverse:
            torch.set_grad_enabled(True)
        else:
            torch.set_grad_enabled(False)

        self.fg_Adjoint = False

        self.u_func = Function(self.V, name="displacement")
        self.v_func = Function(self.V, name="velocity")
        self.a_func = Function(self.V, name="acceleration")
        self.w_func = Function(self.V, name="axilary variable")
        self.H_func = Function(self.V, name="hisory term")
        self.p_func = Function(self.V, name="loading")

        self.u = torch.zeros(self.u_func.vector().get_local().shape, requires_grad=True).double()
        self.v = torch.zeros_like(self.u, requires_grad=True)
        self.a = torch.zeros_like(self.u, requires_grad=True)
        self.a_new = torch.zeros_like(self.u, requires_grad=True)
        self.history = torch.zeros_like(self.u, requires_grad=True)
        self.w = torch.zeros_like(self.u)

        self.kernel.init()

        self.observations = []
        self.Energy_elastic = np.array([])
        self.Energy_kinetic = np.array([])


    def update_forces(self, time):
        if self.NeumannBC:
            self.loading.t = time
            self.f_surf = assemble(self.loading_form)


    def update_state(self):

        h  = self.dt
        un = self.u
        vn = self.v
        an = self.a
        an1= self.a_new

        beta, gamma = self.Newmark.beta, self.Newmark.gamma

        self.u = un + h * vn + 0.5*h**2 * ( (1-2*beta)*an + 2*beta*an1 )        
        self.v = vn + h * ( (1-gamma)*an + gamma*an1 )
        self.a = 1.*an1

        # if self.DirichletBC: self.bc_u.apply(self.u.vector())

        if self.fg_viscosity:
            self.history = self.kernel.update_history(self.v)

            ### auxilary variable is not backpropagated, so the content is mutable
            self.w[:] = ( self.kernel.Weights * self.kernel.modes ).sum(dim=-1)



    def export_state(self, time=0):
        if self.fg_export:
            if not hasattr(self, "file_results"):
                if self.fg_Adjoint:
                    filename = "results"
                else:
                    filename = "results_detached"
                self.file_results = XDMFFile(self.outputfolder+filename+".xdmf")
                self.file_results.parameters["flush_output"] = True
                self.file_results.parameters["functions_share_mesh"] = True


            self.u_func.vector()[:] = self.u.detach().numpy()
            self.v_func.vector()[:] = self.v.detach().numpy()
            self.a_func.vector()[:] = self.a.detach().numpy()
            self.p_func.vector()[:] = self.f_surf

            if self.fg_viscosity:
                self.w_func.vector()[:] = self.w.detach().numpy()
                self.H_func.vector()[:] = self.history.detach().numpy()

            self.file_results.write(self.u_func, time)
            self.file_results.write(self.v_func, time)
            self.file_results.write(self.a_func, time)
            self.file_results.write(self.w_func, time)
            self.file_results.write(self.p_func, time)



    def observe(self):
        if self.observer:
            if not hasattr(self, "observations"): self.observations = []
            obs_n = self.observer.observe()
            self.observations.append(obs_n)

    """
    ==================================================================================================================
    USER DEFINED ROUTINES
    ==================================================================================================================
    """

    def user_defined_routines(self, time=None):

        ### TODO: your code here

        ### EXAMPLE: energies
        if not self.fg_inverse:
            E_elas = assemble(0.5*self.k(self.u_func, self.u_func))
            E_kin  = assemble(0.5*self.m(self.v_func, self.v_func))
            # E_damp += dt*assemble(c(v_old, v_old))
            # E_ext += assemble(Wext(u-u_old))
            # E_tot = E_elas+E_kin #+E_damp #-E_ext
            self.Energy_elastic = np.append(self.Energy_elastic, E_elas)
            self.Energy_kinetic = np.append(self.Energy_kinetic, E_kin)



    """
    ==================================================================================================================
    Forward map
    ==================================================================================================================
    """
    
    def forward_solve(self, parameters=None, record=False, annotate=True, objective=None):

        if parameters is not None:
            self.kernel.update_parameters(parameters)

        self.initialize_state()

        pbar = tqdm(total=self.time_steps.size)

        for (i, t) in enumerate(self.time_steps):

            self.update_forces(t)
            
            self.solve_linear_system()

            self.update_state()
            self.export_state(t)

            self.observe()
            self.user_defined_routines(t)

            pbar.update(1)
        del(pbar)

        self.observations = torch.stack(self.observations)
        return self.observations


    """
    ==================================================================================================================
    Solver via Torch-FEniCS interface
    ==================================================================================================================
    """

    
    def solve_linear_system(self):
        un = self.u.reshape([1, -1, self.ndim])
        vn = self.v.reshape([1, -1, self.ndim])
        an = self.a.reshape([1, -1, self.ndim])
        Hn = self.history.reshape([1, -1, self.ndim])
        coef_a = self.kernel.coef_a.reshape([1, 1])
        coef_c = self.kernel.coef_c.reshape([1, 1])
        a_new = self.__call__(un, vn, an, Hn, coef_a, coef_c)
        self.a_new = a_new.flatten()


    def solve(self, un, vn, an, Hn, coef_a, coef_c):
        h  = self.dt
        beta, gamma = self.Newmark.beta, self.Newmark.gamma

        u_, v_= TrialFunction(self.V), TestFunction(self.V)

        u_star = un + h * vn + 0.5*h**2 * (1-2*beta) * an
        rhs    = self.forces_form - self.k(u_star, v_)
        coef1  = h**2 * beta
        lhs    = self.m(u_, v_) + coef1 * self.k(u_,v_)

        # if self.fg_viscosity:
        u_star_visc = 0.5*h*coef_c * vn + 0.5*h*coef_a * (vn + h*(1-gamma) * an) + Hn
        rhs   = rhs - self.c(u_star_visc, v_)
        coef2 = (0.5 * h**2 * gamma) * coef_a
        lhs = lhs + coef2 * self.c(u_,v_)

        A, b = fenics_adjoint.assemble_system(lhs, rhs, bcs=self.bc_a)

        self.LinSolver.set_operator(A)

        a_new = Function(self.V)

        self.LinSolver.solve(a_new.vector(), b)

        # solve(lhs == rhs, a_new, self.bc_a)

        return a_new


    def input_templates(self):
        return ( Function(self.V), Function(self.V), Function(self.V), Function(self.V), Constant(0.), Constant(0.) )


"""
==================================================================================================================
Newmark container
==================================================================================================================
"""

class Newmark:

    def __init__(self, beta=0.25, gamma=0.5) -> None:
        self.set(beta=beta, gamma=gamma)

    def set(self, beta, gamma):
        self.beta  = beta
        self.gamma = gamma



"""
==================================================================================================================
Default linear solver
==================================================================================================================
"""

def set_linSolver():
	# solver = dl.PETScLUSolver("mumps")
	# solver = dl.PETScKrylovSolver("bicgstab", "amg")
	# solver = dl.PETScKrylovSolver("gmres", "amg")
	# solver = PETScKrylovSolver("cg", "ilu")
	solver = KrylovSolver("cg", "ilu")
	solver.parameters["maximum_iterations"] = 1000
	solver.parameters["relative_tolerance"] = 1.e-6
	solver.parameters["absolute_tolerance"] = 1.e-6
	solver.parameters["error_on_nonconvergence"] = True
	solver.parameters["nonzero_initial_guess"] = False
	solver.parameters["monitor_convergence"] = False
	return solver