

from config import *

tikz_folder = config['outputfolder']


"""
==================================================================================================================
Load data
==================================================================================================================
"""
tip_init, EnergyElastic_init, EnergyKinetic_init, theta_init                        = load_data(config['inputfolder']+"initial_model")
tip_pred, EnergyElastic_pred, EnergyKinetic_pred, theta_pred, convergence_history   = load_data(config['inputfolder']+"inferred_model")
tip_true, EnergyElastic_true, EnergyKinetic_true, theta_true                        = load_data(config['inputfolder']+"target_model")
EnergyTotal_pred = EnergyElastic_pred + EnergyKinetic_pred
EnergyTotal_true = EnergyElastic_true + EnergyKinetic_true

tip_meas = np.loadtxt(config['inputfolder']+"data_tip_displacement_noisy.csv")

time_steps = np.linspace(0, config['FinalTime'], config['nTimeSteps']+1)[1:]
time_steps_meas = time_steps[:tip_meas.size]



"""
==================================================================================================================
Construct kernels
==================================================================================================================
"""

alpha = 0.7
from math import gamma
def kernel_frac(t):
    k = t**(alpha-1) / gamma(alpha)
    return k

alpha_init = 0.5
from math import gamma
def kernel_frac_init(t):
    k = t**(alpha_init-1) / gamma(alpha_init)
    return k

RA = RationalApproximation(alpha=alpha_init)
parameters_init = list(RA.c) + list(RA.d)
if config["infmode"]: parameters_init.append(RA.c_inf)
kernel_init = SumOfExponentialsKernel(parameters = theta_init)


kernel_true = SumOfExponentialsKernel(parameters = theta_true)
#theta_true = np.array(theta_true)
#nModes = len(theta_true) // 2
#c1 = theta_true[:nModes]
#d1 = theta_true[nModes:2*nModes]
#@np.vectorize
#def kernel_exp_true(t):
#    return np.sum(c1 * np.exp(-d1*t))

theta_pred = np.array([i.detach().numpy() for i in theta_pred[0]])
kernel_pred = SumOfExponentialsKernel(parameters = theta_pred)
#c2, d2 = np.array(theta_pred.detach()).reshape([2,-1])
#@np.vectorize
#def kernel_exp_pred(t):
#    return np.sum(c2 * np.exp(-d2*t))




"""
==================================================================================================================
FIGURES
==================================================================================================================
"""

import tikzplotlib
import matplotlib
# plt.style.use("ggplot")
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
    'markersize'   :   2,
}

legend_settings = {
    'loc'             :   'center left',
    'bbox_to_anchor'  :   (1.1, 0.5),
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
    plt.figure('Tip displacement', **figure_settings)
    # plt.title('Tip displacement')
    plt.plot(time_steps, tip_init, "-",  color="gray", label="initial", **plot_settings)
    plt.plot(time_steps, tip_pred, "r-",  label="predict", **plot_settings)
    plt.plot(time_steps, tip_true, "b--", label="truth", **plot_settings)
    plt.plot(time_steps_meas, tip_meas, "ko:", label="data", **plot_settings)
    plt.legend()
    plt.ylabel(r"Tip displacement")
    plt.xlabel(r"$t$")

    tikzplotlib.save(tikz_folder+"plt_tip_displacement.tex", **tikz_settings)


    """
    ==================================================================================================================
    Figure 2: Energies
    ==================================================================================================================
    """
    plt.figure('Energies', **figure_settings)
    # plt.title('Energies')

    plt.plot(time_steps, EnergyElastic_pred, "-", color='red', label="Elastic energy (predict)", **plot_settings)
    plt.plot(time_steps, EnergyKinetic_pred, "-", color='orange', label="Kinetic energy (predict)", **plot_settings)
    # plt.plot(time_steps, EnergyTotal_pred, "-", color='brown', label="Total energy (predict)")

    plt.plot(time_steps, EnergyElastic_true, "--", color='blue', label="Elastic energy (truth)", **plot_settings)
    plt.plot(time_steps, EnergyKinetic_true, "--", color='cyan', label="Kinetic energy (truth)", **plot_settings)
    # plt.plot(time_steps, EnergyTotal_true, "--", color='magenta', label="Total energy (truth)", **plot_settings)

    plt.grid(True, which='both')
    plt.ylabel(r"Energy")
    plt.xlabel(r"$t$")
    plt.legend()

    tikzplotlib.save(tikz_folder+"plt_energies.tex", **tikz_settings)



    """
    ==================================================================================================================
    Figure 3: Kernels
    ==================================================================================================================
    """
    plt.figure('Kernels', **figure_settings)
    # plt.title('Kernels')
    t = np.geomspace(0.04, 4, 100)
    plt.plot(t, kernel_init.eval_func(t), "-", color="gray", label="sum-of-exponentials (initial guess)", **plot_settings)
    plt.plot(t, kernel_pred.eval_func(t), "r-", label="sum-of-exponentials (predict)", **plot_settings)
    plt.plot(t, kernel_true.eval_func(t), "b-", label="sum-of-exponentials (truth)", **plot_settings)
    plt.plot(t, kernel_frac_init(t), "o", color="gray", label=r"fractional $\alpha=0.5$", **plot_settings)
    plt.plot(t, kernel_frac(t), "bo", label=r"fractional $\alpha=0.7$", **plot_settings)
    plt.xscale('log')
    plt.ylabel(r"$k(t)$")
    plt.xlabel(r"$t$")
    plt.legend()

    tikzplotlib.save(tikz_folder+"plt_kernels.tex", **tikz_settings)


    """
    ==================================================================================================================
    Figure 4: Parameters convergence
    ==================================================================================================================
    """
    
    parameters = convergence_history["parameters"]
    p = np.array(parameters)
    nmodes = p.shape[-1]//2
    #nsteps = len(parameters)
    #p = torch.stack(parameters).reshape([nsteps,2,-1]).detach().numpy()

    plt.figure('Parameters convergence: Weights', **figure_settings)
    # plt.title('Parameters convergence: Weights')
    for i in range(nmodes):
        plt.plot(p[:,0,i]/(1+p[:,0,i+nmodes]), label=r'$w_{{%(i)d}}$' % {'i' : i+1}, **plot_settings)
    plt.ylabel(r"$\frac{w}{1+\lambda}$")
    plt.xlabel("Iteration")
    plt.legend()

    tikzplotlib.save(tikz_folder+"plt_weights_convergence.tex", **tikz_settings)
    # plt.yscale('log')


    plt.figure('Parameters convergence: Exponents', **figure_settings)
    # plt.title('Parameters convergence: Exponents')
    for i in range(nmodes):
        plt.plot(p[:,0,i+nmodes]/(1+p[:,0,i+nmodes]), label=r'$\lambda_{{%(i)d}}$' % {'i' : i+1}, **plot_settings)
    # plt.yscale('log')
    plt.ylabel(r"$\frac{\lambda}{1+\lambda}$")
    plt.xlabel("Iteration")
    plt.legend()

    tikzplotlib.save(tikz_folder+"plt_exponents_convergence.tex", **tikz_settings)
    
    if len(p)%2!=0:
        plt.figure('Parameters convergence: Infmode', **figure_settings)
        # plt.title('Parameters convergence: Exponents')
        plt.plot(p[:,0,-1]/(1+p[:,0,-1]), **plot_settings)
        # plt.yscale('log')
        plt.ylabel(r"$w_\infty$")
        plt.xlabel("Iteration")
        plt.legend()

        tikzplotlib.save(tikz_folder+"plt_exponents_convergence.tex", **tikz_settings)


    """
    ==================================================================================================================
    Figure 5: Convergence
    ==================================================================================================================
    """

    loss = convergence_history["loss"]
    grad = convergence_history["grad"]

    plt.figure("Convergence", **figure_settings)
    plt.plot(loss, label="Loss", **plot_settings)
    plt.plot(grad, label="Gradient", **plot_settings)
    plt.legend()
    plt.yscale("log")
    plt.xlabel("Iteration")

    """
    ==================================================================================================================
    """

    plt.show()




