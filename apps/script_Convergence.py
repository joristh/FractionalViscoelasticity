
#%%
from matplotlib.pyplot import figure
from config import *

# smaller mesh for testing
mesh = BoxMesh(Point(0., 0., 0.), Point(1., 0.1, 0.04), 20, 3, 1)
config['mesh'] = mesh

# compute sum of exponentials approximation for fixed alpha
alpha = 1.0
RA = RationalApproximation(alpha=alpha)
config['nModes']    = RA.nModes
config['weights']   = RA.c
config['exponents'] = RA.d

n_steps = [100 * 2**i for i in range(4)]
n_steps.append(n_steps[-1]*5)
result = []

#%%
for n in n_steps:

    config['nTimeSteps'] = n

    Model = ViscoelasticityProblem(**config)
    def Forward():
            Model.forward_solve()
            obs = Model.observations
            return obs.numpy()

    data = Forward()
    result.append(data.flatten())

#%%
reference = result.pop()
n_ref = n_steps.pop()
dt_ref = config['FinalTime']/n_ref

#%%
dt = config['FinalTime']/np.array(n_steps)
stride = n_ref/np.array(n_steps)

error = [np.linalg.norm(reference[::int(stride[i])]-result[i], ord=np.inf) for i in range(len(result))]
ord = np.log(error[-2]/error[-1])/np.log(dt[-2]/dt[-1])
# %%

plt.plot(dt, error, "o")
plt.plot(dt, dt**ord/dt[-1]**ord*error[-1])
plt.xscale("log")
plt.yscale("log")

print("Order of convergence: ", ord)
# %%
