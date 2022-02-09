import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from config import *
from scipy.optimize import curve_fit

plt.style.use("bmh")
font = {'size'   : 12}
matplotlib.rc('font', **font)

alpha = 1.0

dir = config['outputfolder']
dir_plot = dir + "convergence/plots/"
dir += f"convergence/alpha{alpha}/"

sol = []
nsteps  = []

data = {}
numsteps = []
for filename in os.listdir(dir):
    filepath = os.path.join(dir, filename)
    if not os.path.isfile(filepath):
        continue
    tmp_num = int(float((filename.split("_")[-1]).rstrip(".txt")))
    numsteps.append(tmp_num)
    tmp_data = np.insert(np.loadtxt(filepath), 0, 0.)
    data.update({tmp_num : tmp_data})

print("All solutions loaded.")
numsteps = sorted(numsteps)
print(numsteps)
reference_steps = numsteps.pop()
reference = data[reference_steps]

dt = config['FinalTime']/np.array(numsteps)

#Plot solutions
t = np.linspace(0, config['FinalTime'], len(reference))
plotskip = 100
plt.plot(t[::plotskip], reference[::plotskip])
plt.xlabel("Time [s]")
plt.ylabel("Tip displacement [arb. unit]")
plt.title(f"Alpha= {alpha}")
plt.legend()
plt.savefig(dir_plot+f"Solution_{alpha}.pdf", bbox_inches="tight")
plt.show()

#error1 = []
#
#for numstep in numsteps:
#    error1.append(np.abs(data[numstep][-1] - reference[-1]))
#
#order = np.log(error1[-2]/error1[-1])/np.log(dt[-2]/dt[-1])
#print("Order: ", order)
#
#plt.plot(dt, error1, "o-")
##plt.title(f"Convergence  -  Alpha= {alpha}  -  Order= {ord:{0}.{3}}")
#plt.yscale("log")
#plt.xscale("log")
#plt.xlabel("$dt$")
#plt.ylabel("$\mathcal{E}_\infty(dt)$")
#plt.savefig(dir_plot+f"Convergence_{alpha}.pdf", bbox_inches="tight")
#plt.show()


error2 = []

for i, numstep in enumerate(numsteps):
    skip = reference_steps//numstep
    error = 0
    for j in range(numstep):
        u_ref_tip = reference[j*skip]
        u_tip = data[numstep][j]
        error += (u_ref_tip-u_tip)**2
    error2.append(np.sqrt(dt[i]*error))

order = np.log(error2[-2]/error2[-1])/np.log(dt[-2]/dt[-1])
print("Order: ", order)

def f(dt, coeff, order):
    return np.log(dt)*order + coeff

param, param_cov = curve_fit(f, dt[3:], np.log(np.array(error2)[3:]))
fit_error = np.exp(f(dt, param[0], param[1]))
print(param)


plt.plot(dt, error2, "o-", label="Data", zorder=10)
plt.plot(dt, fit_error, label=f"Fit - order = {param[1]:{0}.{3}}", c="k", linestyle="--", zorder=9)
plt.title(f"Convergence Viscoelasticity - Alpha={alpha}")
plt.yscale("log")
plt.xscale("log")
plt.xlabel("$dt$")
plt.ylabel("$\mathcal{E}_{tip}(dt)$")
plt.legend()
plt.savefig(dir_plot+f"Convergence_{alpha}.pdf", bbox_inches="tight")
plt.show()

print("All plots created.")