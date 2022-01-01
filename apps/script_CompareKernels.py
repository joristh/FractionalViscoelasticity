

from config import *

tikz_folder = config['outputfolder']

"""
==================================================================================================================
"""

t = np.geomspace(0.04, 2, 1000)
def l1_misfit(func1, func2):
    y1  = func1(t)
    y2  = func2(t)
    w   = 1 #np.exp(-pi*t)
    h   = np.diff(t)
    err = w*np.abs(y1-y2)
    eps = np.sum((err[1:]+err[:-1])/2 * h)
    return eps


"""
==================================================================================================================
Load data
==================================================================================================================
"""
tip_init, EnergyElastic_init, EnergyKinetic_init, theta_init = load_data(config['inputfolder']+"initial_model")
tip_true, EnergyElastic_true, EnergyKinetic_true, theta_true = load_data(config['inputfolder']+"target_model")

c1, d1 = np.array(theta_true)
@np.vectorize
def kernel_exp_true(t):
    return np.sum(c1 * np.exp(-d1*t))
    # return np.sum(c1 / (t+d1))

alpha_init = 0.5
RA = RationalApproximation(alpha=alpha_init)
c0, d0 = RA.c, RA.d
@np.vectorize
def kernel_exp_init(t):
    return np.sum(c0 * np.exp(-d0*t))
    # return np.sum(c0 / (t+d0))

cases = [
    # 'plots_noise1_M8',
    'plots_noise2_M8',
    'plots_noise4_M8',
    'plots_noise6_M8',
    'plots_noise8_M8',
    # 'plots_noise16_M8',
    # 'plots_noise1_M22',
    # 'plots_noise2_M22',
    # 'plots_noise4_M22',
    # 'plots_noise8_M22',
    # 'plots_noise1_M18',
    # 'plots_noise1_M15',
    # 'plots_noise1_M8',
    ]


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
    # 'loc'             :   'center left',
    # 'bbox_to_anchor'  :   (1.1, 0.5),
}
legend_entries = [
    # r'$1\%$ noise',
    r'$2\%$ noise',
    r'$4\%$ noise',
    r'$6\%$ noise',
    r'$8\%$ noise',
    r'$16\%$ noise',
]

tikz_settings = {
    'axis_width'  :   '0.45\\textwidth',
}




"""
==================================================================================================================
Figure: Kernels
==================================================================================================================
"""
plt.figure('Kernels_log', **figure_settings)
plt.plot(t, kernel_exp_init(t), "-", color="gray", label="initial guess", **plot_settings)
plt.plot(t, kernel_exp_true(t), "b-", label="target", **plot_settings)
plt.xscale('log')
plt.ylabel(r"$k(t)$")
plt.xlabel(r"$t$")


plt.figure('Kernels', **figure_settings)
plt.plot(t, kernel_exp_init(t), "-", color="gray", label="initial guess", **plot_settings)
plt.plot(t, kernel_exp_true(t), "b-", label="target", **plot_settings)
plt.ylabel(r"$k(t)$")
plt.xlabel(r"$t$")


for i, case in enumerate(cases):
    filename = config['inputfolder'] + case + "/inferred_model"
    tip_pred, EnergyElastic_pred, EnergyKinetic_pred, theta_pred, convergence_history = load_data(filename)

    c2, d2 = np.array(theta_pred.square().detach()).reshape([2,-1])
    @np.vectorize
    def kernel_exp_pred(t):
        return np.sum(c2 * np.exp(-d2*t))
        # return np.sum(c2 / (t+d2))

    eps = l1_misfit(kernel_exp_true, kernel_exp_pred)
    print("{0:s}: l1_error = {1:f}".format(case,eps))
    

    plt.figure('Kernels_log', **figure_settings)
    plt.plot(t, kernel_exp_pred(t), "--", label=legend_entries[i], **plot_settings)


    plt.figure('Kernels', **figure_settings)
    plt.plot(t, kernel_exp_pred(t), "--", label=legend_entries[i], **plot_settings)


cN1 = np.array([1.0735e-01, 1.7315e-01, 3.6693e-01, 8.8820e-01, 1.0860e+00, 2.2113e+00, 6.1638e+00, 2.3872e+01])**2
dN1 = np.array([ 1.2458e-03, 1.6176e-02, 9.4062e-02, 5.1814e-01, 2.3980e+00, 1.4902e+01, 9.7315e+01, 8.5205e+02])**2
@np.vectorize
def kernel_noise1(t):
    return np.sum(cN1 * np.exp(-dN1*t))

cN2 = np.array([1.0668e-01, 1.7207e-01, 3.6453e-01, 8.7697e-01, 1.0899e+00, 2.2209e+00, 6.1644e+00, 2.3872e+01])**2
dN2 = np.array([1.2450e-03, 1.6150e-02, 9.3422e-02, 5.0594e-01, 2.4230e+00, 1.4901e+01, 9.7315e+01, 8.5205e+02])**2
@np.vectorize
def kernel_noise2(t):
    return np.sum(cN2 * np.exp(-dN2*t))

plt.figure('Kernels_log', **figure_settings)
# plt.plot(t, kernel_noise1(t), "--", label="noise1 15it", **plot_settings)
# plt.plot(t, kernel_noise2(t), "--", label="noise2 15it", **plot_settings)
plt.legend(**legend_settings)
tikzplotlib.save(tikz_folder+"plt_compare_resulting_kernels_log.tex", **tikz_settings)

plt.figure('Kernels', **figure_settings)
# plt.plot(t, kernel_noise1(t), "--", label="noise1 15it", **plot_settings)
# plt.plot(t, kernel_noise2(t), "--", label="noise2 15it", **plot_settings)
plt.legend(**legend_settings)
tikzplotlib.save(tikz_folder+"plt_compare_resulting_kernels.tex", **tikz_settings)





"""
==================================================================================================================
Figure: Loss convergence
==================================================================================================================
"""
plt.figure('Loss convergence', **figure_settings)

plt.figure('Loss convergence', **figure_settings)
plt.ylabel(r"Loss")
plt.xlabel(r"$iteration$")
plt.yscale('log')


for i, case in enumerate(cases):
    filename = config['inputfolder'] + case + "/inferred_model"
    tip_pred, EnergyElastic_pred, EnergyKinetic_pred, theta_pred, convergence_history = load_data(filename)
    J = convergence_history["loss"]    
    plt.plot(np.arange(1,15), J[:14], 'o-', label=legend_entries[i], **plot_settings)

plt.legend()
tikzplotlib.save(tikz_folder+"plt_loss_convergence.tex", **tikz_settings)

"""
==================================================================================================================
"""

plt.show()