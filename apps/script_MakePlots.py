

from config import *

"""
==================================================================================================================
Load data
==================================================================================================================
"""
tip_pred, EnergyElastic_pred, EnergyKinetic_pred, theta_pred = load_data(config['inputfolder']+"inferred_model")
tip_true, EnergyElastic_true, EnergyKinetic_true, theta_true = load_data(config['inputfolder']+"target_model")
EnergyTotal_pred = EnergyElastic_pred + EnergyKinetic_pred
EnergyTotal_true = EnergyElastic_true + EnergyKinetic_true

tip_meas = np.loadtxt(config['inputfolder']+"data_tip_displacement_noisy.csv")[:20]

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

c1, d1 = np.array(theta_pred.abs().detach()).reshape([2,-1])
@np.vectorize
def kernel_exp_pred(t):
    return np.sum(c1 * np.exp(-d1*t))


c0, d0 = np.array(theta_true)
@np.vectorize
def kernel_exp_true(t):
    return np.sum(c0 * np.exp(-d0*t))



"""
==================================================================================================================
FIGURES
==================================================================================================================
"""

with torch.no_grad():

    """
    ==================================================================================================================
    Figure 1: Observations
    ==================================================================================================================
    """
    plt.figure()
    plt.title('Tip displacement')
    plt.plot(time_steps, tip_pred, "r-",  label="predict")
    plt.plot(time_steps, tip_true, "b--", label="truth")
    plt.plot(time_steps_meas, tip_meas, "ko:", label="data")
    plt.legend()


    """
    ==================================================================================================================
    Figure 2: Energies
    ==================================================================================================================
    """
    plt.figure()
    plt.title('Energies')

    plt.plot(time_steps, EnergyElastic_pred, "-", color='red', label="Elastic energy (predict)")
    plt.plot(time_steps, EnergyKinetic_pred, "-", color='orange', label="Kinetic energy (predict)")
    plt.plot(time_steps, EnergyTotal_pred, "-", color='brown', label="Total energy (predict)")

    plt.plot(time_steps, EnergyElastic_true, "--", color='blue', label="Elastic energy (truth)")
    plt.plot(time_steps, EnergyKinetic_true, "--", color='cyan', label="Kinetic energy (truth)")
    plt.plot(time_steps, EnergyTotal_true, "--", color='magenta', label="Total energy (truth)")

    plt.grid(True, which='both')
    plt.legend()



    """
    ==================================================================================================================
    Figure 3: Kernels
    ==================================================================================================================
    """
    plt.figure()
    plt.title('Kernels')
    t = np.logspace(-2, 3, 100)
    plt.plot(t, kernel_exp_pred(t), "r-", label="sum-of-exponentials (predict)")
    plt.plot(t, kernel_exp_true(t), "b-", label="sum-of-exponentials (truth)")
    plt.plot(t, kernel_frac(t), "k--", label="fractional")
    plt.xscale('log')
    plt.legend()


    plt.show()

    # model.kernel.plot()


