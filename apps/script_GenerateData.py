


from matplotlib.pyplot import figure
from config import *

fg_export = True  ### write results on the disk (True) or only solve (False)
config['export_vtk'] = True


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
config['infmode']   = RA.c_inf

# config['nModes']    = 8
# config['weights']   = [1.0735e-01, 1.7315e-01, 3.6693e-01, 8.8820e-01, 1.0860e+00, 2.2113e+00, 6.1638e+00, 2.3872e+01]
# config['exponents'] = [ 1.2458e-03, 1.6176e-02, 9.4062e-02, 5.1814e-01, 2.3980e+00, 1.4902e+01, 9.7315e+01, 8.5205e+02]
# [1.0735e-01, 1.7315e-01, 3.6693e-01, 8.8820e-01, 1.0860e+00, 2.2113e+00,
#         6.1638e+00, 2.3872e+01, 1.2458e-03, 1.6176e-02, 9.4062e-02, 5.1814e-01,
#         2.3980e+00, 1.4902e+01, 9.7315e+01, 8.5205e+02]

# config['nModes']    = 8
# config['weights']   = [1.0668e-01, 1.7207e-01, 3.6453e-01, 8.7697e-01, 1.0899e+00, 2.2209e+00, 6.1644e+00, 2.3872e+01]
# config['exponents'] = [1.2450e-03, 1.6150e-02, 9.3422e-02, 5.0594e-01, 2.4230e+00, 1.4901e+01, 9.7315e+01, 8.5205e+02]

print()
print()
print("================================")
print("       SUM-OF-EXPONENTIALS")
print("================================")

print("nModes    :", config['nModes'])
print("Exponents :", config['exponents'])
print("Weights   :", config['weights'])
print("Infmode   :", config['infmode'])


### Compare kernels

from math import gamma
def kernel(t):
    k = t**(alpha-1) / gamma(alpha)
    return k

def kernel_exp(t):
    k = RA.appx_ker(t)
    return k

# t = np.logspace(-2, 3, 100)

# plt.figure()
# plt.plot(t, kernel_exp(t), "r-", label="sum-of-exponentials")
# plt.plot(t, kernel(t), "b--", label="fractional")
# plt.xscale('log')
# plt.legend()
# plt.show()



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

Model = ViscoelasticityProblem(**config)

loading = config.get("loading", None)
if isinstance(loading, list): ### multiple loadings case
    def Forward():
        obs = torch.tensor([])
        for loading_instance in loading:
            Model.forward_solve(loading=loading_instance)
            obs = torch.cat([obs, Model.observations])
        return obs.numpy()
else:
    def Forward():
        Model.forward_solve()
        obs = Model.observations
        return obs.numpy()

data = Forward()

if fg_export: ### write data to file
    # data = model.observations.numpy()
    np.savetxt(config['outputfolder']+"data_tip_displacement.csv", data)
    save_data(config['outputfolder']+"target_model", Model, other=[[config['weights'], config['exponents'], config['infmode']]])


"""
==================================================================================================================
Display
==================================================================================================================
"""

with torch.no_grad():
    plt.subplot(1,2,1)
    plt.title('Tip displacement')
    plt.plot(Model.time_steps, Model.observations)

    if not Model.fg_inverse:
        plt.subplot(1,2,2)
        plt.title('Energies')
        plt.plot(Model.time_steps, Model.Energy_elastic, "o-", color='blue', label="Elastic energy")
        plt.plot(Model.time_steps, Model.Energy_kinetic, "o-", color='orange', label="Kinetic energy")
        plt.plot(Model.time_steps, Model.Energy_elastic+Model.Energy_kinetic, "o-", color='red', label="Total energy")
        plt.grid(True, which='both')
        plt.legend()

    plt.show()

    # model.kernel.plot()


