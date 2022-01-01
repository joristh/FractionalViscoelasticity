

from config import *
from math import gamma

tikz_folder = config['outputfolder']


"""
==================================================================================================================
Load data
==================================================================================================================
"""
tip_init, EnergyElastic_init, EnergyKinetic_init, theta_init              = load_data(config['inputfolder']+"initial_model")
tip_pred, EnergyElastic_pred, EnergyKinetic_pred, theta_pred, convergence = load_data(config['inputfolder']+"inferred_model")
tip_true, EnergyElastic_true, EnergyKinetic_true, theta_true              = load_data(config['inputfolder']+"target_model")
EnergyTotal_pred = EnergyElastic_pred + EnergyKinetic_pred
EnergyTotal_true = EnergyElastic_true + EnergyKinetic_true

tip_true = np.loadtxt(config['inputfolder']+"data_tip_displacement.csv")
tip_meas = np.loadtxt(config['inputfolder']+"data_tip_displacement_noisy.csv")
tip_init = np.loadtxt(config['inputfolder']+"tip_displacement_init.csv")
tip_pred = np.loadtxt(config['inputfolder']+"tip_displacement_pred.csv")

time_steps = np.linspace(0, config['FinalTime'], config['nTimeSteps']+1)[1:]
time_steps_meas = time_steps[:tip_meas.shape[0]]

tip_true_tr, tip_true_dev = tip_true[...,0], tip_true[...,1]
tip_meas_tr, tip_meas_dev = tip_meas[...,0], tip_meas[...,1]
tip_init_tr, tip_init_dev = tip_init[...,0], tip_init[...,1]
tip_pred_tr, tip_pred_dev = tip_pred[...,0], tip_pred[...,1]



"""
==================================================================================================================
Construct kernels
==================================================================================================================
"""

alpha1, alpha2 = 0.9, 0.7
alpha_init = 0.5

RA = RationalApproximation(alpha=alpha_init, tol=1.e-4)
parameters0 = list(RA.c) + list(RA.d)
kernel_init = SumOfExponentialsKernel(parameters=parameters0)

parameters1, parameters2 = np.array(theta_pred).tolist()
kernel_pred_tr  = SumOfExponentialsKernel(parameters=parameters1)
kernel_pred_dev = SumOfExponentialsKernel(parameters=parameters2)

parameters1, parameters2 = theta_true
kernel_true_tr  = SumOfExponentialsKernel(parameters=parameters1)
kernel_true_dev = SumOfExponentialsKernel(parameters=parameters2)





"""
==================================================================================================================
FIGURES
==================================================================================================================
"""

import tikzplotlib
import matplotlib
plt.style.use("bmh")
font = {
    # 'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 12}
matplotlib.rc('font', **font)


figure_settings = {
    'figsize'   :   (10,6),
}

plot_settings = {
    'markersize'   :   1.3,
}

legend_settings = {
    # 'loc'             :   'center left',
    # 'bbox_to_anchor'  :   (1.1, 0.5),
}


tikz_settings = {
    'axis_width'  :   '\\textwidth',
}



with torch.no_grad():

    """
    ==================================================================================================================
    Figure 1: Observations
    ==================================================================================================================
    """
    plt.figure('Tip displacement (tr)', **figure_settings)
    # plt.title('Tip displacement')
    plt.plot(time_steps, tip_init_tr, "-",  color="gray", label="initial", **plot_settings)
    plt.plot(time_steps, tip_pred_tr, "r-",  label="predict", **plot_settings)
    plt.plot(time_steps, tip_true_tr, "b--", label="truth", **plot_settings)
    plt.plot(time_steps_meas, tip_meas_tr, "ko:", label="data", **plot_settings)
    plt.legend()
    plt.ylabel(r"Tip displacement")
    plt.xlabel(r"$t$")
    plt.gca().yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.2f}')) # 2 decimal places

    tikzplotlib.save(tikz_folder+"plt_two_kernels_tip_displacement_bending.tex", **tikz_settings)


    plt.figure('Tip displacement (dev)', **figure_settings)
    # plt.title('Tip displacement')
    plt.plot(time_steps, tip_init_dev, "-",  color="gray", label="initial", **plot_settings)
    plt.plot(time_steps, tip_pred_dev, "r-",  label="predict", **plot_settings)
    plt.plot(time_steps, tip_true_dev, "b--", label="truth", **plot_settings)
    plt.plot(time_steps_meas, tip_meas_dev, "ko:", label="data", **plot_settings)
    plt.legend()
    plt.ylabel(r"Tip displacement")
    plt.xlabel(r"$t$")

    tikzplotlib.save(tikz_folder+"plt_two_kernels_tip_displacement_extension.tex", **tikz_settings)


    """
    ==================================================================================================================
    Figure 2: Energies
    ==================================================================================================================
    """
    # plt.figure('Energies', **figure_settings)
    # # plt.title('Energies')

    # plt.plot(time_steps, EnergyElastic_pred, "-", color='red', label="Elastic energy (predict)", **plot_settings)
    # plt.plot(time_steps, EnergyKinetic_pred, "-", color='orange', label="Kinetic energy (predict)", **plot_settings)
    # # plt.plot(time_steps, EnergyTotal_pred, "-", color='brown', label="Total energy (predict)")

    # plt.plot(time_steps, EnergyElastic_true, "--", color='blue', label="Elastic energy (truth)", **plot_settings)
    # plt.plot(time_steps, EnergyKinetic_true, "--", color='cyan', label="Kinetic energy (truth)", **plot_settings)
    # # plt.plot(time_steps, EnergyTotal_true, "--", color='magenta', label="Total energy (truth)", **plot_settings)

    # plt.grid(True, which='both')
    # plt.ylabel(r"Energy")
    # plt.xlabel(r"$t$")
    # plt.legend(**legend_settings)

    # tikzplotlib.save(tikz_folder+"plt_energies.tex")



    """
    ==================================================================================================================
    Figure 3: Kernels
    ==================================================================================================================
    """
    plt.figure('Kernels (tr)', **figure_settings)
    # plt.title('Kernels')
    t = np.geomspace(0.04, 2, 100)
    plt.plot(t, kernel_init.eval_func(t), "-", color="gray", label="initial guess", **plot_settings)
    plt.plot(t, kernel_pred_tr.eval_func(t), "r-", label="predict", **plot_settings)
    plt.plot(t, kernel_true_tr.eval_func(t), "b-", label="truth", **plot_settings)
    # plt.plot(t, kernel_frac_init(t), "o", color="gray", label=r"fractional $\alpha=0.5$", **plot_settings)
    # plt.plot(t, t**(alpha1-1)/gamma(alpha1), "bo", label=r"fractional $\alpha=0.9$", **plot_settings)
    plt.xscale('log')
    plt.ylabel(r"$k(t)$")
    plt.xlabel(r"$t$")
    plt.legend(**legend_settings)

    tikzplotlib.save(tikz_folder+"plt_two_kernels_compare_kernels_tr.tex", **tikz_settings)



    plt.figure('Kernels (dev)', **figure_settings)
    # plt.title('Kernels')
    t = np.geomspace(0.04, 2, 100)
    plt.plot(t, kernel_init.eval_func(t), "-", color="gray", label="initial guess", **plot_settings)
    plt.plot(t, kernel_pred_dev.eval_func(t), "r-", label="predict", **plot_settings)
    plt.plot(t, kernel_true_dev.eval_func(t), "b-", label="truth", **plot_settings)
    # plt.plot(t, kernel_frac_init(t), "o", color="gray", label=r"fractional $\alpha=0.5$", **plot_settings)
    # plt.plot(t, kernel_frac(t), "bo", label=r"fractional $\alpha=0.7$", **plot_settings)
    # plt.plot(t, t**(alpha2-1)/gamma(alpha2), "bo", label=r"fractional $\alpha=0.7$", **plot_settings)
    plt.xscale('log')
    plt.ylabel(r"$k(t)$")
    plt.xlabel(r"$t$")
    plt.legend(**legend_settings)

    tikzplotlib.save(tikz_folder+"plt_two_kernels_compare_kernels_dev.tex", **tikz_settings)


    """
    ==================================================================================================================
    """

    plt.show()




