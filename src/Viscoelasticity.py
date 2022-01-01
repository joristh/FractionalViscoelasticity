
from fenics import *
from fenics_adjoint import *
# from numpy_adjoint import *
# import torch_fenics
import fenics_adjoint

import numpy as np

import torch
from torch import nn

from tqdm import tqdm
import matplotlib.pyplot as plt

from .Kernels import SumOfExponentialsKernel
# from .reduced_function import ReducedFunctionTorch




"""
==================================================================================================================
Elastoplasticity forward problem class
==================================================================================================================
"""

class ViscoelasticityProblem(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        self.verbose      = kwargs.get("verbose", False)
        self.fg_export    = kwargs.get("export", None)
        self.inputfolder  = kwargs.get("inputfolder", "./")
        self.outputfolder = kwargs.get("outputfolder", "./")

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



        mesh = self.set_mesh(**kwargs)



        deg_u = kwargs.get("degree", 1)
        V = VectorFunctionSpace(mesh, "CG", deg_u)
        self.V = V

        u  = Function(V, name="displacement")
        v  = Function(V, name="velocity")
        a  = Function(V, name="acceleration")
        a_new = Function(V, name="new acceleration")
        inc= Function(V, name="Newton increment")
        w  = Function(V, name="axilary variable")
        w_tr  = Function(V, name="axilary variable (trace)")
        w_dev = Function(V, name="axilary variable (deviator)")
        
        self.u, self.v, self.a = u, v, a
        self.a_new = a_new
        self.w = w


        self.set_boundary_condition(**kwargs)


        ### Assembling the FE matrices
        u_, v_  = TrialFunction(V), TestFunction(V)
        self.M_form = rho * inner(u_, v_)*dx
        self.K_form = inner(self.sigma(self.eps(u_)), self.eps(v_))*dx
        self.M = assemble(self.M_form, annotate=True)
        self.K = assemble(self.K_form, annotate=True)

        self.fg_viscosity = kwargs.get("viscosity", False)
        self.K_visc = self.K


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
        self.kernel = SumOfExponentialsKernel(**kwargs)
        self.kernel.compute_coefficients(self.dt)


        ### History term
        self.history = Vector(self.u.vector())
        self.history[:] = self.kernel.update_history(self.v.vector().get_local()).detach().numpy()

        ### Linear solver
        self.LinSolver = set_linSolver()

        ### Observations and QoIs
        observation = kwargs.get("observation", None)
        self.observation = lambda : observation(self)


        if self.fg_viscosity:
            self.ViscousTerm = ViscousTerm(self.V, self.dt, **kwargs)


        
        

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

    # def F_ext(self, v):
    #     n, ds = self.n, self.ds
    #     return self.loading*dot(n, v)*ds(4)

    def eps(self, v):
        e = sym(grad(v))
        return e
        # return as_tensor([[e[0, 0], e[0, 1], 0],
        #                 [e[0, 1], e[1, 1], 0],
        #                 [0, 0, 0]])

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

    def solve_detach(self):
        self.fg_Adjoint = False

        self.initialize_state()         

        for (i, t) in tqdm(enumerate(self.time_steps), total=self.time_steps.size):

            self.update_forces(t)
            
            self.update_linear_system()
            self.LinSolver.solve(self.a_new.vector(), self.rhs, annotate=False)

            self.update_fields()
            self.export_fields(t)

            self.user_defined_routines(t)

        # fenics_func = ReducedFunctionTorch(self.u.vector().get_local(), Control(self.kernel.parameters()))

        return self.u



    # def solve_torch(self):
    #     self.fg_Adjoint = True

    #     self.initialize_state()         

    #     for (i, t) in tqdm(enumerate(self.time_steps), total=self.time_steps.size):

    #         self.update_forces(t)
            
    #         self.solve_linear_system()

    #         self.update_fields()
    #         self.export_fields(t)

    #         self.user_defined_routines(t)


    #     return self.u



    def solve(self, parameters=None):
        self.fg_Adjoint = True

        if self.fg_viscosity and (parameters is not None):
            self.ViscousTerm.update_parameters(parameters)
            print("parameters = ", self.ViscousTerm.parameters())

        self.initialize_state()


        for (i, t) in enumerate(self.time_steps):

            self.update_forces(t)

            self.assemble_linear_system()            
            self.LinSolver.solve(self.a_new.vector(), self.rhs)

            self.update_fields()
            self.export_fields(t)

            self.user_defined_routines(t)

            # J = assemble(inner(self.u, self.u)*dx)
            # print("J=", J)

        return self.QoIs


    def input_templates(self):
        return [ Constant(0.)  for k in range(2*self.kernel.nModes) ]


    """
    ==================================================================================================================
    """

    def initialize_state(self):
        # self.u.vector()[:] = 0.
        # self.v.vector()[:] = 0.
        # self.a.vector()[:] = 0.

        self.u  = Function(self.V, name="displacement")
        self.v  = Function(self.V, name="velocity")
        self.a  = Function(self.V, name="acceleration")
        self.a_new = Function(self.V, name="new acceleration")
        

        self.history[:] = 0.
        if hasattr(self.kernel, "modes"): self.kernel.modes[:] = 0.
        self.kernel.compute_coefficients(self.dt)

        if self.fg_viscosity and self.fg_Adjoint:
            self.ViscousTerm.initialize_state()

        self.QoIs = []
        self.Energy_elastic = np.array([])
        self.Energy_kinetic = np.array([])


    def update_forces(self, time):
        if self.NeumannBC:
            self.loading.t = time
            self.f_surf = assemble(self.loading_form)


    def update_linear_system(self):  ### for detached

        h  = self.dt
        un = self.u.vector()
        vn = self.v.vector()
        an = self.a.vector()

        beta, gamma = self.Newmark.beta, self.Newmark.gamma

        coef1 = h**2 * beta
        A     = self.M + coef1 * self.K
        rhs   = self.f_vol + self.f_surf - self.K * (un + h * vn + 0.5*h**2 * (1-2*beta) * an)

        if self.fg_viscosity:
            a, c  = self.kernel.coef_a.detach().numpy(), self.kernel.coef_c.detach().numpy()
            coef2 = 0.5 * h**2 * gamma * a
            A     = A + coef2 * self.K_visc
            rhs   = rhs - self.K_visc * (0.5*h*c * vn + 0.5*h*a * (vn + h*(1-gamma) * an) + self.history)

        if self.DirichletBC: self.bc_a.apply(A, rhs)

        self.LinSolver.set_operator(A)
        self.A, self.rhs = A, rhs



    # def assemble_matrix(self):

    #     beta, gamma = self.Newmark.beta, self.Newmark.gamma

    #     h  = self.dt
    #     un = self.u
    #     vn = self.v
    #     an = self.a

    #     a_ = TrialFunction(self.V)
    #     v_ = TestFunction(self.V)

    #     u_star = un + h * vn + 0.5*h**2 * (1-2*beta) * an
    #     rhs    = self.forces_form - self.k(u_star, v_)
    #     coef1  = h**2 * beta
    #     bilinear_form = self.m(a_, v_) + coef1 * self.k(a_,v_)


    def assemble_linear_system(self):   ### for adjoint

        beta, gamma = self.Newmark.beta, self.Newmark.gamma

        h  = self.dt
        un = self.u
        vn = self.v
        an = self.a

        a_ = TrialFunction(self.V)
        v_ = TestFunction(self.V)

        u_star = un + h * vn + 0.5*h**2 * (1-2*beta) * an
        rhs    = self.forces_form - self.k(u_star, v_)
        coef1  = h**2 * beta
        bilinear_form = self.m(a_, v_) + coef1 * self.k(a_,v_)

        if self.fg_viscosity:
            a, c = Constant(self.ViscousTerm.coef_a), Constant(self.ViscousTerm.coef_c)
            u_star_visc = 0.5*h*c * vn + 0.5*h*a * (vn + h*(1-gamma) * an) + self.ViscousTerm.history
            rhs   = rhs - self.c(u_star_visc, v_)
            coef2 = (0.5 * h**2 * gamma) * a
            bilinear_form = bilinear_form + coef2 * self.c(a_,v_)

        A, b = fenics_adjoint.assemble_system(bilinear_form, rhs, bcs=self.bc_a)

        self.LinSolver.set_operator(A)
        self.A, self.rhs = A, b



    def update_fields(self):

        if self.fg_Adjoint:

            h  = self.dt
            un = self.u
            vn = self.v
            an = self.a
            an1= self.a_new

            beta, gamma = self.Newmark.beta, self.Newmark.gamma

            self.u.assign( un + h * vn + 0.5*h**2 * ( (1-2*beta)*an + 2*beta*an1 ), annotate=True )    
            self.v.assign( vn + h * ( (1-gamma)*an + gamma*an1 ), annotate=True   )
            self.a.assign( an1, annotate=True )

            # u_new  = un + h * vn + 0.5*h**2 * ( (1-2*beta)*an + 2*beta*an1 ) 
            # self.u = Function(self.V)
            # self.u.assign( u_new  )

            # v_new  = vn + h * ( (1-gamma)*an + gamma*an1 )
            # self.v = Function(self.V)
            # self.v.assign( v_new )
            
            # self.a = Function(self.V)
            # self.a.assign( an1 )

            if self.fg_viscosity:
                self.ViscousTerm.update_history(self.v)

        else:
            h  = self.dt
            un = self.u.vector().get_local()
            vn = self.v.vector().get_local()
            an = self.a.vector().get_local()
            an1= self.a_new.vector().get_local()

            beta, gamma = self.Newmark.beta, self.Newmark.gamma

            self.u.vector()[:] = un + h * vn + 0.5*h**2 * ( (1-2*beta)*an + 2*beta*an1 )        
            self.v.vector()[:] = vn + h * ( (1-gamma)*an + gamma*an1 )
            self.a.vector()[:] = an1

            # if self.DirichletBC: self.bc_u.apply(self.u.vector())

            self.history[:] = self.kernel.update_history(self.v.vector().get_local()).detach().numpy()

            self.w.vector()[:] = ( self.kernel.Weights * self.kernel.modes ).sum(dim=-1).detach().numpy()



    def export_fields(self, time=0):
        if self.fg_export:
            if not hasattr(self, "file_results"):
                if self.fg_Adjoint:
                    filename = "results"
                else:
                    filename = "results_detached"
                self.file_results = XDMFFile(self.outputfolder+filename+".xdmf")
                self.file_results.parameters["flush_output"] = True
                self.file_results.parameters["functions_share_mesh"] = True

            self.file_results.write(self.u, time)
            self.file_results.write(self.v, time)
            self.file_results.write(self.a, time)
            self.file_results.write(self.w, time)

            p = Function(self.V, name="loading")
            p.vector()[:] = self.f_surf
            self.file_results.write(p, time)

    """
    ==================================================================================================================
    USER DEFINED ROUTINES
    ==================================================================================================================
    """

    def user_defined_routines(self, time=None):
        un = self.u.vector()
        vn = self.v.vector()
        an = self.a.vector()

        ### TODO: your code here

        if self.observation:
            if not hasattr(self, "QoIs"): self.QoIs = []
            qn = self.observation()
            self.QoIs.append(qn)


        E_elas = assemble(0.5*self.k(self.u, self.u))
        E_kin  = assemble(0.5*self.m(self.v, self.v))
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
    
    def forward(self, parameters, record=False, annotate=True, objective=None):
        # self.kernel.update_parameters(parameters)
        self.solve(parameters)
        return self.QoIs


    """
    ==================================================================================================================
    Torch-FEniCS interface
    ==================================================================================================================
    """

    
    # def solve_linear_system(self):

    #         un = self.torch_u
    #         vn = self.torch_v
    #         an = self.torch_a

    #         self.a_new = self(self.parameters, un, vn, an)


    # def assemble_linear_system_TFI(self):

    #     beta, gamma = self.Newmark.beta, self.Newmark.gamma

    #     h  = self.dt
    #     un = self.u
    #     vn = self.v
    #     an = self.a

    #     a_ = TrialFunction(self.V)
    #     v_ = TestFunction(self.V)

    #     u_star = un + h * vn + 0.5*h**2 * (1-2*beta) * an
    #     rhs    = self.forces_form - self.k(u_star, v_)
    #     coef1  = h**2 * beta
    #     bilinear_form = self.m(a_, v_) + coef1 * self.k(a_,v_)

    #     if self.fg_viscosity:
    #         a, c = Constant(self.ViscousTerm.coef_a), Constant(self.ViscousTerm.coef_c)
    #         u_star_visc = 0.5*h*c * vn + 0.5*h*a * (vn + h*(1-gamma) * an) + self.ViscousTerm.history
    #         rhs   = rhs - self.c(u_star_visc, v_)
    #         coef2 = (0.5 * h**2 * gamma) * a
    #         bilinear_form = bilinear_form + coef2 * self.c(a_,v_)

    #     A, b = fenics_adjoint.assemble_system(bilinear_form, rhs, bcs=self.bc_a)

    #     self.LinSolver.set_operator(A)
    #     self.A, self.rhs = A, b


    # def solve(parameters, un, vn, an):
    #     pass

    # def input_templates(self):
    #     return (Constant(0.), ) * self.kernel.nModes, Function(self.V), Function(self.V), Function(self.V)

"""
==================================================================================================================
Newmark container
==================================================================================================================
"""

class ViscousTerm:

    def __init__(self, V, dt, **kwargs) -> None:
        self.V = V
        self.h = dt

        self.nModes    = kwargs.get("nModes", 1)
        weights        = kwargs.get("weights",   [ 1. ] * self.nModes )
        exponents      = kwargs.get("exponents", [ 0. ] * self.nModes )
        self.weights   = [ AdjFloat(x) for x in weights ]
        self.exponents = [ AdjFloat(x) for x in exponents ]
        # self.set_weights(weights)
        # self.set_exponents(exponents)

        self.initialize_state()

        self.compute_coefficients(self.h)

    
    def parameters(self):
        # p = [0.]*(2*self.nModes)
        # for k in range(self.nModes):
        #     p[k]               = float(self.weights[k])
        #     p[self.nModes + k] = float(self.exponents[k])
        return self.weights + self.exponents


    def update_parameters(self, parameters):
        self.set_weights(parameters[:self.nModes])
        self.set_exponents(parameters[self.nModes:])
        # self.weights[:]   = [ w**2 for w in parameters[:self.nModes] ]
        # self.exponents[:] = [ lmbda**2 for lmbda in parameters[self.nModes:] ]
        self.compute_coefficients(self.h)

    def set_weights(self, weights):
        self.weights[:] = [ (w**2)**0.5 for w in weights ]

    def set_exponents(self, exponents):
        self.exponents[:] = [ (lmbda**2)**0.5 for lmbda in exponents ]



    def compute_coefficients(self, h, gamma=1):
        self.coef_ak = []
        self.coef_bk = []
        self.coef_ck = []
        self.coef_a  = 0.
        self.coef_c  = 0.
        for k in range(self.nModes):
            lmbda = self.exponents[k]
            theta = lmbda / (1 + lmbda)
            lgh   = lmbda*gamma*h
            den   = (1-theta)*(1 + lgh) + theta * h/2 * (1 + 2*lgh)
            ak    = (1 + 2*lgh) / den
            bk    = ( (1-theta)*(1+lgh) - theta * h/2 ) / den
            ck    = AdjFloat(1.) / den
            self.coef_ak.append(ak)
            self.coef_bk.append(bk)
            self.coef_ck.append(ck)
            self.coef_a += self.weights[k] * ak
            self.coef_c += self.weights[k] * ck

        # lmbda = self.exponents
        # theta = lmbda / (1 + lmbda)
        # lgh   = lmbda*gamma*h
        # den   = (1-theta)*(1 + lgh) + theta * h/2 * (1 + 2*lgh)
        # self.coef_ak = (1 + 2*lgh) / den
        # self.coef_bk = ( (1-theta)*(1+lgh) - theta * h/2 ) / den
        # self.coef_ck = 1 / den
        # self.coef_a  = ( self.weights * self.coef_ak ).sum()
        # self.coef_c  = ( self.weights * self.coef_ck ).sum()


    def update_history(self, v):
        h = self.h

        # self.history.vector()[:] = 0.
        # self.history = Function(self.V)
        # self.history.vector[:] = 0.

        for k in range(self.nModes):
            ak, bk, ck = Constant(self.coef_ak[k]), Constant(self.coef_bk[k]), Constant(self.coef_ck[k])
            # wk = self.coef_bk[k] * self.modes[k] + 0.5*h*self.coef_ck[k] * self.v_old + 0.5*h*self.coef_ak[k] * v
            # self.modes[k].assign( self.coef_bk[k] * self.modes[k] + 0.5*h*self.coef_ck[k] * self.v_old + 0.5*h*self.coef_ak[k] * v )
            self.modes[k] = bk * self.modes[k] + 0.5*h*ck * self.v_old + 0.5*h*ak * v
            # new_history = self.history + self.weights[k] * self.coef_bk[k] * self.modes[k]
            # self.history.assign( new_history )

        # self.history = Function(self.V)
        # self.history.assign( sum([self.weights[k] * self.coef_bk[k] * self.modes[k] for k in range(self.nModes)]), annotate=True )
        self.history = sum( [  Constant(self.weights[k] * self.coef_bk[k]) * self.modes[k] for k in range(self.nModes) ] )

        self.v_old.assign(v, annotate=True)


    def initialize_state(self):
        self.modes   = [ Function(self.V) for k in range(self.nModes) ]
        self.history = Function(self.V)
        self.v_old   = Function(self.V)


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
	solver = PETScLUSolver("mumps")
	# solver = dl.PETScKrylovSolver("bicgstab", "amg")
	# solver = dl.PETScKrylovSolver("gmres", "amg")
	# solver = PETScKrylovSolver("cg", "ilu")
	# solver = KrylovSolver("cg", "ilu")
	# solver = KrylovSolver("cg", "hypre_euclid")
	# solver.parameters["maximum_iterations"] = 1000
	# solver.parameters["relative_tolerance"] = 1.e-6
	# solver.parameters["absolute_tolerance"] = 1.e-6
	# solver.parameters["error_on_nonconvergence"] = True
	# solver.parameters["nonzero_initial_guess"] = False
	# solver.parameters["monitor_convergence"] = False
	return solver