import numpy as np
import sys
import os
from config import *
import time

# smaller mesh for faster execution
mesh = BoxMesh(Point(0., 0., 0.), Point(1., 0.1, 0.04), 15, 2, 1)
config['mesh'] = mesh

# get input values
alpha = float(sys.argv[1])
order = int(sys.argv[2])

# compute sum of exponentials approximation for fixed alpha
RA = RationalApproximation(alpha=alpha)
config['nModes']    = RA.nModes
config['weights']   = RA.c
config['exponents'] = RA.d
path = config['outputfolder']

n_steps = 10**order
config['nTimeSteps'] = n_steps

#print(f"START: dt={1/n_steps} started")

time.sleep(.5)
Model = ViscoelasticityProblem(**config)
Model.forward_solve()
obs = Model.observations
data = obs.numpy()
path = path+f"convergence/alpha{alpha}/"
os.makedirs(path, exist_ok=True)
np.savetxt(path+f"nsteps{n_steps}.txt", data)

print(f"END: dt={1/n_steps} finished")