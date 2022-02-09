import numpy as np
import sys
import os
from config import *
import time
from datetime import datetime

# smaller mesh for faster execution
mesh = BoxMesh(Point(0., 0., 0.), Point(1., 0.1, 0.04), 20, 4, 2)
config['mesh'] = mesh

# get input values
alpha = float(sys.argv[1])
index = int(sys.argv[2])
maxindex = int(sys.argv[3])

timestring = datetime.strftime(datetime.now(), "%Y%m%d%H%M")

# infmode boolean from config
infmode = config.get('infmode', False)

# compute sum of exponentials approximation for fixed alpha
RA = RationalApproximation(alpha=alpha)
parameters = list(RA.c) + list(RA.d)
if infmode==True: parameters.append(RA.c_inf)
kernel  = SumOfExponentialsKernel(parameters=parameters)
kernels = [kernel]

path = config['outputfolder']

config['viscosity'] = True

n_steps_list = 2**np.arange(0, maxindex)*1e2
n_steps_list = np.append(n_steps_list, n_steps_list[-1]*10)
n_steps = n_steps_list[index]
config['nTimeSteps'] = int(n_steps*config['FinalTime'])

print(f"START: dt={1/n_steps} started")

Model = ViscoelasticityProblem(**config, kernels=kernels)
Model.forward_solve(loading=config["loading"][0])
obs = Model.observations
data = obs.numpy()
path = path+f"convergence/alpha{alpha}/"
os.makedirs(path, exist_ok=True)
np.savetxt(path+f"tipdisplacement_{timestring}_{alpha}_{n_steps}.txt", data)

print(f"END: dt={1/n_steps} finished")