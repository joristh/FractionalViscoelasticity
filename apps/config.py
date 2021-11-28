
from fenics import *
from fenics_adjoint import *

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.Viscoelasticity import ViscoelasticityProblem
from src.Kernels import SumOfExponentialsKernel
from src.InverseProblem import InverseProblem
from src.Observers import TipDisplacementObserver
from src.Objectives import MSE
from src.Regularization import myRegularizationTerm as reg
from src.RationalApproximation import RationalApproximation_AAA as RationalApproximation
from src.data_manager import save_data, load_data


"""
==================================================================================================================
Problem Configuration
==================================================================================================================
"""

inputfolder  = "./workfolder/"
outputfolder = "./workfolder/"

### Beam
mesh = BoxMesh(Point(0., 0., 0.), Point(1., 0.1, 0.04), 60, 10, 5)

### Sub domain for clamp at left end
def DirichletBoundary(x, on_boundary):
    return near(x[0], 0.) and on_boundary

### Sub domain for rotation at right end
def NeumannBoundary(x, on_boundary):
    return near(x[0], 1.) and on_boundary

### loading (depending on t)
cutoff_time    = 4/5
magnitude      = 1.
load_Bending   = Expression(("0", "t <= tc ? p0*t/tc : 0", "0"), t=0, tc=cutoff_time, p0=magnitude, degree=0) ### Bending
magnitude      = 1.e2
load_Extension = Expression(("t <= tc ? p0*t/tc : 0", "0", "0"), t=0, tc=cutoff_time, p0=magnitude, degree=0) ### Extension


config = {
    'verbose'           :   True,
    'inputfolder'       :   inputfolder,
    'outputfolder'      :   outputfolder,
    'export_vtk'        :   False,

    'FinalTime'         :   4,
    'nTimeSteps'        :   100,

    'mesh'              :   mesh,
    'DirichletBoundary' :   DirichletBoundary,
    'NeumannBoundary'   :   NeumannBoundary,
    'loading'           :   [load_Bending, load_Extension], ###  load_Bending, [load_Bending, load_Extension]

    ### Material parameters
    'Young'             :   1.e3,
    'Poisson'           :   0.3,
    'density'           :   1.,

    ### Viscous term
    'viscosity'         :   True,
    'two_kernels'       :   True,

    ### Measurements
    'observer'          :   TipDisplacementObserver,
    'noise_level'       :   2, ### [%]

    ### Optimization
    'optimizer'         :   torch.optim.LBFGS, ### E.g., torch.optim.SGD, torch.optim.LBFGS (recommended), ...
    'nepochs'           :   10,
    'tol'               :   1.e-5,
    'regularization'    :   None,  ### your regularization function, e.g., "reg", or None/False for no regularization
    'initial_guess'     :   None,  ### initial guess for parameters calibration: (weights, exponents)
}