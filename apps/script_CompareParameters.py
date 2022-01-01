

from config import *

tikz_folder = config['outputfolder']


"""
==================================================================================================================
Load data
==================================================================================================================
"""

tip_pred, EnergyElastic_pred, EnergyKinetic_pred, theta_pred, convergence_history1   = load_data(config['inputfolder'] + "plots_noise1_M8/" + "inferred_model")
tip_pred, EnergyElastic_pred, EnergyKinetic_pred, theta_pred, convergence_history2   = load_data(config['inputfolder'] + "plots_noise2_M8/" + "inferred_model")




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
    'axis_width'  :   '0.5\\textwidth',
}



with torch.no_grad():





    """
    ==================================================================================================================
    Figure: Parameters convergence
    ==================================================================================================================
    """

    parameters1 = convergence_history1["parameters"]
    nsteps1 = len(parameters1)
    p1 = torch.stack(parameters1).square().reshape([nsteps1,2,-1]).detach().numpy()
    J1 = convergence_history1["loss"]

    parameters2 = convergence_history2["parameters"]
    nsteps2 = len(parameters2)
    p2 = torch.stack(parameters2).square().reshape([nsteps2,2,-1]).detach().numpy()
    J2 = convergence_history2["loss"]

    plt.figure('Parameters convergence: Weights', **figure_settings)
    # plt.title('Parameters convergence: Weights')
    for i in range(p1.shape[-1]):
        plt.plot(p1[:,0,i]/(1+p1[:,1,i]), label=r'$w_{{%(i)d}}$' % {'i' : i+1}, **plot_settings)
        plt.plot(p2[:,0,i]/(1+p2[:,1,i]), '--', label=r'$w_{{%(i)d}}$' % {'i' : i+1}, **plot_settings)
    plt.ylabel(r"$\frac{w}{1+\lambda}$")
    plt.xlabel(r"$iteration$")
    plt.legend(**legend_settings)

    tikzplotlib.save(tikz_folder+"plt_weights_convergence.tex", **tikz_settings)
    # plt.yscale('log')


    plt.figure('Parameters convergence: Exponents', **figure_settings)
    # plt.title('Parameters convergence: Exponents')
    for i in range(p1.shape[-1]):
        plt.plot(p1[:,1,i]/(1+p1[:,1,i]), label=r'$\lambda_{{%(i)d}}$' % {'i' : i+1}, **plot_settings)
        plt.plot(p2[:,1,i]/(1+p2[:,1,i]), '--', label=r'$\lambda_{{%(i)d}}$' % {'i' : i+1}, **plot_settings)
    # plt.yscale('log')
    plt.ylabel(r"$\frac{\lambda}{1+\lambda}$")
    plt.xlabel(r"$iteration$")
    plt.legend(**legend_settings)

    tikzplotlib.save(tikz_folder+"plt_exponents_convergence.tex", **tikz_settings)

    """
    ==================================================================================================================
    Figure: Loss convergence
    ==================================================================================================================
    """

    plt.figure('Loss convergence', **figure_settings)
    plt.plot(J1, label=r'$1\%$', **plot_settings)
    plt.plot(J2, label=r'$2\%$', **plot_settings)
    plt.ylabel(r"Loss")
    plt.xlabel(r"$iteration$")
    plt.yscale('log')
    plt.legend()


    """
    ==================================================================================================================
    """

    plt.show()




