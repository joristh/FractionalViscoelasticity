
from config import *

fg_export = True    ### write results on the disk (True) or only solve (False)


"""
==================================================================================================================
Initial guess
==================================================================================================================
"""

alpha = 0.5
RA = RationalApproximation(alpha=alpha)
config['nModes'] = RA.nModes
config['initial_guess'] = [RA.c, RA.d]


"""
==================================================================================================================
Data to fit
==================================================================================================================
"""

data_true = np.loadtxt(config['inputfolder']+"data_tip_displacement.csv")
data = data_true.copy()

### Optimize on a shorter interval
data = data[:int(data.size//2)]
T, nsteps = config['FinalTime'], config['nTimeSteps']
config['nTimeSteps'] = data.size
config['FinalTime']  = data.size * (T / nsteps)

### Noisy data
noise = np.random.normal(loc=0, scale=1.e-2*np.abs(data).max(), size=data.shape) ### additive noise
data  = data + noise
np.savetxt(config['outputfolder']+"data_tip_displacement_noisy.csv", data)


### Compare data
plt.figure()
plt.plot(data_true, "r-", label="true data")
plt.plot(data, "bo--", label="measurements")
plt.legend()
plt.show()




"""
==================================================================================================================
Inverse problem
==================================================================================================================
"""

print()
print()
print("================================")
print("       INVERSE PROBLEM")
print("================================")


model = ViscoelasticityProblem(**config)

objective = MSE(data=data)
IP        = InverseProblem(**config)

theta_opt = IP.calibrate(model, objective, **config)

print("Optimal parameters :", theta_opt)
print("Final loss         :", IP.loss)




"""
==================================================================================================================
Forward run of the inferred model
==================================================================================================================
"""

print()
print()
print("================================")
print("       RUN RESULTING MODEL")
print("================================")

### Recover the original time settings
model.set_time_stepper(nTimeSteps=nsteps, FinalTime=T)

model.forward_solve()

if fg_export: ### write data to file
    save_data(config['outputfolder']+"inferred_model", model, other=[theta_opt])


"""
==================================================================================================================
Display
==================================================================================================================
"""

with torch.no_grad():
    plt.subplot(1,2,1)
    plt.title('Tip displacement')
    plt.plot(model.time_steps, model.observations, "r-",  label="prediction")
    plt.plot(model.time_steps, data_true, "b--", label="truth")
    plt.plot(model.time_steps[:data.size], data, "bo", label="data")
    plt.legend()

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

