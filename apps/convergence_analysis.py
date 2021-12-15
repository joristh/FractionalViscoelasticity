import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from config import *

plt.style.use("bmh")
font = {'size'   : 12}
matplotlib.rc('font', **font)

alpha = 1.
exclude_loading = False

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

if exclude_loading:
    for i, solution in enumerate(sol):
        sol[i] = solution[len(solution)//2:]
    nsteps = nsteps/2

t = np.linspace(0, config['FinalTime'], 1000)
stride_list = []
for i in range(0, len(nsteps)):
    if i==0:
        stride = int(nsteps[i]/100)
        plt.plot(t[::10], sol[i][::stride], label=f"dt = {dt[i]}")
    else:
        stride = int(nsteps[i]/1000)
        plt.plot(t, sol[i][::stride], label=f"dt = {dt[i]}")
    stride_list.append(int(nsteps[i]/100))

plt.xlabel("Time [s]")
plt.ylabel("Tip displacement [arb. unit]")
plt.title(f"Alpha= {alpha}")
plt.legend()
plt.savefig(dir_plot+f"Solution_{alpha}.pdf", bbox_inches="tight")
plt.show()

reference = sol[-1][::stride_list[-1]]
err = []
for i in range(len(sol)-1):
    err.append(np.linalg.norm(sol[i][::stride_list[i]]-reference, ord=np.inf))

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