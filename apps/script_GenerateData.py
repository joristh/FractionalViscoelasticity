


from matplotlib.pyplot import figure
from config import *

fg_export = True  ### write results on the disk (True) or only solve (False)
config['export_vtk'] = True


"""
==================================================================================================================
Kernel and its rational approximation
==================================================================================================================
"""

if config['two_kernels']:
    alpha1 = 0.9
    RA = RationalApproximation(alpha=alpha1)
    parameters1 = list(RA.c) + list(RA.d)
    kernel1 = SumOfExponentialsKernel(parameters=parameters1)

    alpha2 = 0.7
    RA = RationalApproximation(alpha=alpha2)
    parameters2 = list(RA.c) + list(RA.d)
    kernel2 = SumOfExponentialsKernel(parameters=parameters2)

    kernels    = [kernel1, kernel2]
    parameters = [parameters1, parameters2]

else:
    alpha = 0.7
    RA = RationalApproximation(alpha=alpha)
    parameters = list(RA.c) + list(RA.d)
    kernel  = SumOfExponentialsKernel(parameters=parameters)
    kernels = [kernel]





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

Model = ViscoelasticityProblem(**config, kernels=kernels)

loading = config.get("loading", None)
if isinstance(loading, list): ### multiple loadings case
    def Forward():
        obs = torch.tensor([])
        for loading_instance in loading:
            Model.forward_solve(loading=loading_instance)
            obs = torch.cat([obs, Model.observations], dim=-1)
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
    save_data(config['outputfolder']+"target_model", Model, other=[parameters])


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
        # plt.plot(Model.time_steps, Model.Energy_elastic+Model.Energy_kinetic, "o-", color='red', label="Total energy")
        plt.grid(True, which='both')
        plt.legend()

    plt.show()

    # model.kernel.plot()


