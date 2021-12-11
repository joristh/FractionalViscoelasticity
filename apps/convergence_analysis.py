import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from config import *

plt.style.use("bmh")
font = {'size'   : 12}
matplotlib.rc('font', **font)

alpha = 1.

dir = config['outputfolder']
dir_plot = dir + "convergence/plots/"
dir += f"convergence/alpha{alpha}/"

sol = []
nsteps  = []

for filename in os.listdir(dir):
   
   sol.append(np.loadtxt(dir+filename))
   nsteps.append(len(sol[-1]))
   #print(len(sol[-1]))

print("All solutions loaded.")

sol, nsteps = np.array(sol, dtype=object), np.array(nsteps)
idx = np.argsort(nsteps)
sol, nsteps = sol[idx], nsteps[idx]

dt = config['FinalTime']/nsteps

t = np.linspace(0, config['FinalTime'], 100)
stride = []
for i in range(len(nsteps)):
    stride.append(int(nsteps[i]/100))
    plt.plot(t, sol[i][::stride[i]], label=f"dt = {dt[i]}")

plt.xlabel("Time [s]")
plt.ylabel("Tip displacement [arb. unit]")
plt.title(f"Alpha= {alpha}")
plt.legend()
plt.savefig(dir_plot+f"Solution_{alpha}.pdf", bbox_inches="tight")
plt.show()

reference = sol[-1][::stride[-1]]
err = []
for i in range(len(sol)-1):
    err.append(np.linalg.norm(sol[i][::stride[i]]-reference, ord=np.inf))

ord = np.log(err[-2]/err[-1])/np.log(dt[-3]/dt[-2])

plt.plot(dt[:-1], err, "o-")
plt.title(f"Convergence  -  Alpha= {alpha}  -  Order= {ord:{0}.{3}}")
plt.yscale("log")
plt.xscale("log")
plt.xlabel("$dt$")
plt.ylabel("$\mathcal{E}_\infty(dt)$")
plt.savefig(dir_plot+f"Convergence_{alpha}.pdf", bbox_inches="tight")
plt.show()

print("All plots created.")