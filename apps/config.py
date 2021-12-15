
from fenics import *
from fenics_adjoint import *

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.Viscoelasticity_torch import ViscoelasticityProblem
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
mesh = BoxMesh(Point(0., 0., 0.), Point(1., 0.1, 0.04), 20, 2, 1)

### Sub domain for clamp at left end
def DirichletBoundary(x, on_boundary):
    return near(x[0], 0.) and on_boundary

### Sub domain for excitation at right end
def NeumannBoundary(x, on_boundary):
    return near(x[0], 1.) and on_boundary

### loading (depending on t)
magnitude   = 1.
cutoff_time = 4/5
load_Bending   = Expression(("0", "t <= tc ? p0*t/tc : 0", "0"), t=0, tc=cutoff_time, p0=magnitude, degree=0) ### Bending
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
    'loading'           :   load_Bending, ###  load_Bending, [load_Extension, load_Bending] ### default loading (mainly for initialization)

    ### Material parameters
    'Young'             :   1.e3,
    'Poisson'           :   0.3,
    'density'           :   1.,

    ### Viscous term
    'viscosity'         :   True,
    'nModes'            :   None,
    'weights'           :   None,
    'exponents'         :   None,
    'split'             :   False, ### split kernels into hydrostatic and deviatoric parts

    ### Measurements
    'observer'          :   TipDisplacementObserver,
    'noise_level'       :   6, ### [%]

    ### Optimization
    'optimizer'         :   torch.optim.LBFGS, ### E.g., torch.optim.SGD, torch.optim.LBFGS, ...
    'max_iter'          :   100,
    'tol'               :   1.e-5,
    'regularization'    :   None,  ### your regularization function, e.g., "reg", or None/False for no regularization
    'initial_guess'     :   None,  ### initial guess for parameters calibration: (weights, exponents)
    'line_search_fn'    :   'strong_wolfe', ### None, 'strong_wolfe',
}