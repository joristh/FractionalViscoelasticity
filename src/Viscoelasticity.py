
from fenics import *
from fenics_adjoint import *
# from numpy_adjoint import *
import torch_fenics
import fenics_adjoint

import numpy as np

import torch
from torch import nn

import dill as pickle

from tqdm import tqdm
import matplotlib.pyplot as plt

from .Kernels import SumOfExponentialsKernel


"""
==================================================================================================================
Elastoplasticity forward problem class
==================================================================================================================
"""

class ViscoelasticityProblem(torch_fenics.FEniCSModule):

    def __init__(self, **kwargs):
        super().__init__()

        # self.verbose      = kwargs.get("verbose", False)
        # self.fg_export_vtk= kwargs.get("export_vtk", None)
        # self.inputfolder  = kwargs.get("inputfolder", "./")
        # self.outputfolder = kwargs.get("outputfolder", "./")
        # self.fg_viscosity = kwargs.get("viscosity", False)
        # self.fg_inverse   = kwargs.get("InverseProblem", False)

        self.flags = {
            'verbose'       :   kwargs.get("verbose", False),
            'inverse'       :   kwargs.get("InverseProblem", False),
            'viscosity'     :   kwargs.get("viscosity", False),
            'export_vtk'    :   kwargs.get("export_vtk", None)
        }

        ### Elastic parameters
        E  = kwargs.get('Young', 1.e3)
        nu = kwargs.get('Poisson', 0.3)
        E, nu = Constant(E), Constant(nu)
        lmbda = E*nu/(1+nu)/(1-2*nu)
        mu    = E/2./(1+nu)
        self.lmbda, self.mu = lmbda, mu

        ### Mass density
        rho = kwargs.get('density', 1.)
        rho = Constant(rho)
        self.rho = rho


        ### Mesh
        mesh = self.set_mesh(**kwargs)

        ### Finite elements space
        deg_u = kwargs.get("degree", 1)
        V = VectorFunctionSpace(mesh, "CG", deg_u)
        self.V = V
        u_, v_ = TrialFunction(V), TestFunction(V)

        ### Boundary conditions
        self.set_boundary_condition(**kwargs)
        
        ### Source terms
        # self.set_load(**kwargs)
        
        ### Time scheme
        self.Newmark = Newmark()
        self.set_time_stepper(**kwargs)

        ### Integral kernels
        self.set_kernels(**kwargs)        

        ### Linear solver
        self.LinSolver = set_linSolver()

        ### Observations and QoIs
        observer = kwargs.get("observer", None)
        if observer:
            self.observer = observer(Model=self)


    ### Set the kernels
    def set_kernels(self, **kwargs):
        kernels = kwargs.get("kernels", SumOfExponentialsKernel(**kwargs))
        if kernels:
            if not isinstance(kernels, list): kernels = [kernels]
            self.flags['unique_kernel'] = (len(kernels)==1)
            self.kernels = nn.ModuleList(kernels)
        else:
            self.flags['viscosity'] = False
            if self.flags['verbose']:
                print("Empty kernel: viscosity term will be ignored.")


    ### Set the source terms
    def set_load(self, **kwargs):
        u_, v_  = TrialFunction(self.V), TestFunction(self.V)

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
    Mechanical tensors calculus
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


    def c_tr(self, u, u_):
        return inner((self.lmbda+2./self.ndim*self.mu)*tr(self.eps(u))*Identity(3), self.eps(u_))*dx

    def c_dev(self, u, u_):
        return inner(2*self.mu*dev(self.eps(u)), self.eps(u_))*dx

    """
    ==================================================================================================================
    """

    def initialize_state(self):

        if self.flags['inverse']:
            torch.set_grad_enabled(True)
        else:
            torch.set_grad_enabled(False)

        #TODO: write the output FEniCS function as dictionary "state"
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

        if self.flags['viscosity']:
            for kernel in self.kernels: kernel.init(h=self.dt)
            self.history = [ torch.zeros_like(self.u, requires_grad=True) for kernel in self.kernels ]
            # self.w = torch.zeros_like(self.u)

        self.observations   = []
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

        if self.flags['viscosity']:
            self.history = [ kernel.update_history(self.v) for kernel in self.kernels ]

            ### auxilary variable is not backpropagated, so the content is mutable
            # self.w[:] = ( self.kernel.Weights * self.kernel.modes ).sum(dim=-1)

        ### Update FEniCS state functions (if needed)
        if (not self.flags['inverse']) or self.flags['export_vtk']:
            self.u_func.vector()[:] = self.u.detach().numpy()
            self.v_func.vector()[:] = self.v.detach().numpy()
            self.a_func.vector()[:] = self.a.detach().numpy()
            self.p_func.vector()[:] = self.f_surf
            # if self.flag['viscosity']:
            #     self.w_func.vector()[:] = self.w.detach().numpy()
            #     self.H_func.vector()[:] = self.history.detach().numpy()



    def export_state(self, time=0):
        if self.flags['export_vtk']:
            if not hasattr(self, "file_results"):
                filename = "results"
                self.file_results = XDMFFile(self.outputfolder+filename+".xdmf")
                self.file_results.parameters["flush_output"] = True
                self.file_results.parameters["functions_share_mesh"] = True

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
        if not self.flags['inverse']:
            E_elas = assemble(0.5*self.k(self.u_func, self.u_func))
            E_kin  = assemble(0.5*self.m(self.v_func, self.v_func))
            self.Energy_elastic = np.append(self.Energy_elastic, E_elas)
            self.Energy_kinetic = np.append(self.Energy_kinetic, E_kin)



    """
    ==================================================================================================================
    Forward map
    ==================================================================================================================
    """
    
    def forward_solve(self, loading=None):

        if loading is not None:
            self.set_load(loading=loading)

        self.initialize_state()

        for (i, t) in tqdm(enumerate(self.time_steps), total=self.time_steps.size):

            self.update_forces(t)
            
            self.solve_linear_system()

            self.update_state()
            self.export_state(t)

            self.observe()
            self.user_defined_routines(t)
            

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

        ### Hydrostatic viscousity term
        Hn         = self.history[0].reshape([1, -1, self.ndim])
        coef_a     = self.kernels[0].coef_a.reshape([1, 1])
        coef_c     = self.kernels[0].coef_c.reshape([1, 1])

        ### Deviatoric viscousity term
        i = 0 if self.flags['unique_kernel'] else 1
        Hn_dev     = self.history[i].reshape([1, -1, self.ndim])
        coef_a_dev = self.kernels[i].coef_a.reshape([1, 1])
        coef_c_dev = self.kernels[i].coef_c.reshape([1, 1])

        a_new = self.__call__(un, vn, an, Hn, coef_a, coef_c, Hn_dev, coef_a_dev, coef_c_dev)

        self.a_new = a_new.flatten()


    def solve(self, un, vn, an, Hn, coef_a, coef_c, Hn_dev, coef_a_dev, coef_c_dev):
        h  = self.dt
        beta, gamma = self.Newmark.beta, self.Newmark.gamma

        u_, v_= TrialFunction(self.V), TestFunction(self.V)

        ### Elastic part
        u_star = un + h * vn + 0.5*h**2 * (1-2*beta) * an
        rhs    = self.forces_form - self.k(u_star, v_)
        coef1  = h**2 * beta
        lhs    = self.m(u_, v_) + coef1 * self.k(u_,v_)

        ### Viscous part
        if self.flags['viscosity']:
            ### Hydrostatic term
            u_star_visc = 0.5*h*coef_c * vn + 0.5*h*coef_a * (vn + h*(1-gamma) * an) + Hn
            rhs   = rhs - self.c_tr(u_star_visc, v_)
            coef2 = (0.5 * h**2 * gamma) * coef_a
            lhs   = lhs + coef2 * self.c_tr(u_,v_)

            ### Deviatioric term                
            u_star_visc = 0.5*h*coef_c_dev * vn + 0.5*h*coef_a_dev * (vn + h*(1-gamma) * an) + Hn_dev
            rhs   = rhs - self.c_dev(u_star_visc, v_)
            coef2 = (0.5 * h**2 * gamma) * coef_a_dev
            lhs   = lhs + coef2 * self.c_dev(u_,v_)

        A, b = fenics_adjoint.assemble_system(lhs, rhs, bcs=self.bc_a)

        self.LinSolver.set_operator(A)

        a_new = Function(self.V)

        self.LinSolver.solve(a_new.vector(), b)

        return a_new


    def input_templates(self):
        return (    Function(self.V), Function(self.V), Function(self.V),
                    Function(self.V), Constant(0.), Constant(0.), 
                    Function(self.V), Constant(0.), Constant(0.)    )


    # """
    # ==================================================================================================================
    # Save and load the object
    # ==================================================================================================================
    # """

    # def save(self, filename): ### filename = full/relative path w/o extension
    #     if not filename.endswith('.pkl'):
    #         filename = filename + '.pkl'

    #     with open(filename, 'wb') as filehandler:
    #         data = [self.observations, self.Energy_elastic, self.Energy_kinetic]
    #         pickle.dump(data, filehandler)

    #     if self.verbose:
    #         print("Object data is saved to {0:s}".format(filename))


    # def load(self, filename):
    #     if not filename.endswith('.pkl'):
    #         filename = filename + '.pkl'

    #     with open(filename, 'rb') as filehandler:
    #         data = pickle.load(filehandler)
    #         self.observations, self.Energy_elastic, self.Energy_kinetic = data




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