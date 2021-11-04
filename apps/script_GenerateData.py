


from matplotlib.pyplot import figure
from config import *

fg_export = True  ### write results on the disk (True) or only solve (False)



"""
==================================================================================================================
Kernel and its rational approximation
==================================================================================================================
"""

alpha = 0.7
RA = RationalApproximation(alpha=alpha)
config['nModes']    = RA.nModes
config['weights']   = RA.c
config['exponents'] = RA.d


print()
print()
print("================================")
print("       SUM-OF-EXPONENTIALS")
print("================================")

print("nModes    :", config['nModes'])
print("Exponents :", config['exponents'])
print("Weights   :", config['weights'])


### Compare kernels

from math import gamma
def kernel(t):
    k = t**(alpha-1) / gamma(alpha)
    return k

def kernel_exp(t):
    k = RA.appx_ker(t)
    return k

t = np.logspace(-2, 3, 100)

plt.figure()
plt.plot(t, kernel_exp(t), "r-", label="sum-of-exponentials")
plt.plot(t, kernel(t), "b--", label="fractional")
plt.xscale('log')
plt.legend()
plt.show()



"""
==================================================================================================================
Forward problem for generating data
==================================================================================================================
"""

print()
print()
print("================================")
print("       FORWARD RUN")
print("================================")

model = ViscoelasticityProblem(**config)

model.forward_solve()

if fg_export: ### write data to file
    data = model.observations.numpy()
    np.savetxt(config['outputfolder']+"data_tip_displacement.csv", data)
    save_data(config['outputfolder']+"target_model", model, other=[[config['weights'], config['exponents']]])


"""
==================================================================================================================
Display
==================================================================================================================
"""

with torch.no_grad():
    plt.subplot(1,2,1)
    plt.title('Tip displacement')
    plt.plot(model.time_steps, model.observations)

    if not model.fg_inverse:
        plt.subplot(1,2,2)
        plt.title('Energies')
        plt.plot(model.time_steps, model.Energy_elastic, "o-", color='blue', label="Elastic energy")
        plt.plot(model.time_steps, model.Energy_kinetic, "o-", color='orange', label="Kinetic energy")
        plt.plot(model.time_steps, model.Energy_elastic+model.Energy_kinetic, "o-", color='red', label="Total energy")
        plt.grid(True, which='both')
        plt.legend()

    plt.show()

    # model.kernel.plot()


