


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
    # parameters1 = np.array([2.79058303e-02, 3.84100129e-02, 5.89650116e-02, 3.37202484e-02,
    #    3.40912830e-01, 1.28161585e+00, 2.23960854e+00, 4.80126425e+00,
    #    3.68392812e-02, 1.32895882e-01, 2.93308726e-01, 6.40031339e-01,
    #    2.13387389e+00, 3.90843676e+00, 9.89557763e+00, 2.92032700e+01])**2
    kernel1 = SumOfExponentialsKernel(parameters=parameters1)

    alpha2 = 0.7
    RA = RationalApproximation(alpha=alpha2)
    parameters2 = list(RA.c) + list(RA.d)
    # parameters2 = np.array([ 0.32337598,  0.41615834,  0.47182692,  1.03023015,  0.24184555,
    #     0.85714041, -0.11852263,  3.39125769,  0.10000886,  0.26773023,
    #     0.12313697,  0.70114342,  1.4451129 ,  3.89811345, 10.0704782 ,
    #    29.40964287])**2
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

    # np.savetxt(config['outputfolder']+"tip_displacement_init.csv", data)
    # save_data(config['outputfolder']+"initial_model", Model, other=[parameters])


    # np.savetxt(config['outputfolder']+"tip_displacement_pred.csv", data)
    # save_data(config['outputfolder']+"inferred_model", Model, other=[parameters])


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


