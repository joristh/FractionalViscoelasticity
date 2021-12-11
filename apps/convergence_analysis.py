import os
import numpy as np
import matplotlib.pyplot as plt
from config import *

dir = config['outputfolder']
dir += "convergence/alpha1.0/"

sol = []
nsteps  = []

for filename in os.listdir(dir):
   
   sol.append(np.loadtxt(dir+filename))
   nsteps.append(len(sol[-1]))
   print(len(sol[-1]))

sol, nsteps = np.array(sol, dtype=object), np.array(nsteps)
idx = np.argsort(nsteps)
sol, nsteps = sol[idx], nsteps[idx]

dt = config['FinalTime']/nsteps

t = np.linspace(0, config['FinalTime'], 100)
stride = []
for i in range(len(nsteps)):
    stride.append(int(nsteps[i]/100))
    plt.plot(t, sol[i][::stride[i]], label=f"dt = {dt[i]}")

plt.legend()
plt.show()


print(dt)