

from fenics import *
from fenics_adjoint import *
import fenics

import numpy as np
import matplotlib.pyplot as plt

from src.Viscoelasticity_torch import ViscoelasticityProblem
from src.InverseProblem import InverseProblem
from src.Observations import TipDisplacementObserver







"""
==================================================================================================================
Configuration
==================================================================================================================
"""

inputfolder  = "./workfolder/"
outputfolder = "./workfolder/"


### Beam
mesh = BoxMesh(Point(0., 0., 0.), Point(1., 0.1, 0.04), 60, 10, 5)
# mesh = RectangleMesh.create([p0,p1],[nx,ny],CellType.Type.quadrilateral)


# Sub domain for clamp at left end
def DirichletBoundary(x, on_boundary):
    return near(x[0], 0.) and on_boundary

# Sub domain for rotation at right end
def NeumannBoundary(x, on_boundary):
    return near(x[0], 1.) and on_boundary

### loading (depending on t)
p0 = 1.
cutoff_Tc = 4/5
loading = Expression(("0", "t <= tc ? p0*t/tc : 0", "0"), t=0, tc=cutoff_Tc, p0=p0, degree=0)
# loading = Expression(("0", "0", "t <= tc ? p0*t/tc : 0"), t=0, tc=cutoff_Tc, p0=p0, degree=0)

# ### Observation operator: here tip displacement
# def observation(Model): 
#     # # u_tip = Model.u(1., 0.05, 0.02)[1]
#     # u  = Model.u ### no torch
#     u  = Model.u_func ### torch
#     ds = Model.ds_Neumann
#     if not hasattr(Model, "tip_area"):
#         Model.tip_area = assemble(1.*ds)
#         # Model.tip_form = Model.u.sub(1)*ds
#     # u_tip = [ assemble(u.sub(j)*ds, annotate=True) / S for j in range(Model.ndim) ]
#     u_tip = assemble(u.sub(1)*ds, annotate=False) / Model.tip_area
#     # J = assemble(inner(u,u)*dx)
#     # print('Tip displacemetn = ', u_tip)
#     return u_tip

### Loss function
def myLossFunction(Model, parameters, data):
    y = Model.forward_solve(parameters)
    assert( len(y) == len(data) )
    J = 0.
    for n, yn in enumerate(y):
        J += (yn - data[n])**2
    # dJ = compute_gradient(J, [Control(p) for p in Model.ViscousTerm.parameters()] )
    # print("\n -> My monitor:")
    # print("              Objective = ", J)
    # print("         grad Objective = ", dJ)
    # print()
    # J = assemble(inner(y,y)*dx)
    return J





config = {
    'verbose'           :   True,
    'inputfolder'       :   inputfolder,
    'outputfolder'      :   outputfolder,
    'export'            :   True,
    'mode'              :   "forward", ### "generate_data", "inverse", "forward"

    'FinalTime'         :   4,
    'nTimeSteps'        :   40,

    'mesh'              :   mesh,
    'DirichletBoundary' :   DirichletBoundary,
    'NeumannBoundary'   :   NeumannBoundary,
    'loading'           :   loading,

    'Young'             :   1.e3,
    'Poisson'           :   0.3,
    'density'           :   1.,

    'viscosity'         :   True,
    'nModes'            :   1,
    'weights'           :   [1.], #, 10], #[100,  100, 100, 100],
    'exponents'         :   [0.], #, 1000], #[0.1,  1, 10, 100],

    'observer'          :   TipDisplacementObserver,
    # 'observation'       :   observation,
    'objective'         :   myLossFunction,
}





"""
==================================================================================================================
Inverse problem
==================================================================================================================
"""

### NOTE: Forward problem works calling solve_detach()
### NOTE: Inverse problem has yet to be debugged and optimized

model = ViscoelasticityProblem(**config)

if config['mode'] in ("forward", "generate_data"): ### only forward run
    # model.solve_detach()
    # model.solve_torch()
    # model.solve([0.08055463539336585, 9.999962036183991])
    # model.solve()
    model.forward_solve()

    if config['mode'] == "generate_data": ### write data to file
        data = np.array(model.QoIs)
        np.savetxt(outputfolder+"data_tip_displacement.csv", data)

elif config['mode'] == "inverse": ### Inverse problem
    print("================================")
    print("       INVERSE PROBLEM")
    print("================================")

    data = np.loadtxt(inputfolder+"data_tip_displacement.csv")
    IP   = InverseProblem(**config)

    theta_ini = [0.01, 0.1, 1., 10.]
    theta_opt = IP.calibrate(model, data, initial_guess=theta_ini)
    print("Optimal parameters :", theta_opt)
    # print("Final objective :", IP.final_objective)



"""
==================================================================================================================
Output
==================================================================================================================
"""
plt.subplot(1,2,1)
plt.title('Tip displacement')
plt.plot(model.time_steps, model.QoIs)

plt.subplot(1,2,2)
plt.title('Energies')
plt.plot(model.time_steps, model.Energy_elastic, "o-", color='blue', label="Elastic energy")
plt.plot(model.time_steps, model.Energy_kinetic, "o-", color='orange', label="Elastic kinetic")
plt.plot(model.time_steps, model.Energy_elastic+model.Energy_kinetic, "o-", color='red', label="Total energy")
plt.grid(True, which='both')
plt.legend()

plt.show()

# model.kernel.plot()


