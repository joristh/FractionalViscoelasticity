

from fenics import *
from fenics_adjoint import *
import fenics

import numpy as np
import matplotlib.pyplot as plt

from src.Viscoelasticity_torch import ViscoelasticityProblem
from src.InverseProblem import InverseProblem
from src.Observers import TipDisplacementObserver
from src.Objectives import MSE







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



config = {
    'verbose'           :   True,
    'inputfolder'       :   inputfolder,
    'outputfolder'      :   outputfolder,
    'export'            :   True,
    'mode'              :   "inverse", ### "generate_data", "inverse", "forward"

    'FinalTime'         :   4,
    'nTimeSteps'        :   10,

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

    ### Optimization
    'observer'          :   TipDisplacementObserver,
    'nepochs'           :   100,
}





"""
==================================================================================================================
Inverse problem
==================================================================================================================
"""

### NOTE: Forward problem works calling solve_detach() for "pyadjoint" version and "forward_solve" for torch version
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

    data      = np.loadtxt(inputfolder+"data_tip_displacement.csv")
    objective = MSE(data=data)
    IP        = InverseProblem(**config)

    theta_ini = [0.01, 0.1, 1., 10.]
    theta_opt = IP.calibrate(model, objective=objective, initial_guess=theta_ini, **config)
    print("Optimal parameters :", theta_opt)
    print("Final objective :", IP.loss)



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


