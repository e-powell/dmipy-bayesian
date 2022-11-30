# #TO DO
# OUTPUT GIBBS MCMC MOVES
# INCORPORATE TQDM
# CHECK THE LOWEST POSSIBLE NSTEPS WITHOUT FAILING
# PLOT ORIENTATION PARAMETERS

# load some necessary modules
from dmipy.core import modeling_framework
from os.path import join
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
# load HCP acqusition scheme
from dmipy.data import saved_acquisition_schemes
acq_scheme = saved_acquisition_schemes.wu_minn_hcp_acquisition_scheme()
import asyncio
from importlib import reload
import fit_bayes
from fit_bayes import fit, tform_params  # , dict_to_array, array_to_dict
fit_bayes = reload(fit_bayes)


# set up a simple GM/WM/CSF image

# bigger image
dimx=25
dimy=25
# 0 is WM, 1 is GM, 2 is CSF
simROIs = np.zeros((dimx, dimy))
simROIs[0:10, :] = 0    # WM
simROIs[10:20, :] = 1   # GM
simROIs[20:25, :] = 2   # CSF
simmask = np.ones((dimx, dimy))



#set up NODDI model

from dmipy.signal_models import cylinder_models, gaussian_models
ball = gaussian_models.G1Ball()
stick = cylinder_models.C1Stick()
zeppelin = gaussian_models.G2Zeppelin()
from dmipy.distributions.distribute_models import SD1WatsonDistributed
watson_dispersed_bundle = SD1WatsonDistributed(models=[stick, zeppelin])
watson_dispersed_bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp','C1Stick_1_lambda_par','partial_volume_0')
watson_dispersed_bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
#fix the diffusivities
watson_dispersed_bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par', 1.7e-9)
from dmipy.core.modeling_framework import MultiCompartmentModel
NODDI_mod = MultiCompartmentModel(models=[ball, watson_dispersed_bundle])
NODDI_mod.set_fixed_parameter('G1Ball_1_lambda_iso', 3e-9)
NODDI_mod.parameter_ranges
# choose some parameters for the simulated NODDI model
mu = (np.pi, 0)
SD1WatsonDistributed_1_SD1Watson_1_odi = [0.2, 0.6, 0.7]
SD1WatsonDistributed_1_partial_volume_0 = [0.8, 0.6, 0.9]
partial_volume_0 = [0.1, 0.1, 0.9]
partial_volume_1 = [1 - x for x in partial_volume_0]
# lambda_par = 1.7e-9  # in m^2/s
# lambda_iso = 3e-9  # in m^2/s
# f_0 = 0.3
# f_1 = 0.7
parameter_vector = NODDI_mod.parameters_to_parameter_vector(
    SD1WatsonDistributed_1_SD1Watson_1_mu=mu,
    SD1WatsonDistributed_1_SD1Watson_1_odi=SD1WatsonDistributed_1_SD1Watson_1_odi,
    SD1WatsonDistributed_1_partial_volume_0=SD1WatsonDistributed_1_partial_volume_0,
    partial_volume_0=partial_volume_0,
    partial_volume_1=partial_volume_1)
# simulate a single voxel for each ROI
nmeas = len(acq_scheme.bvalues)
E = NODDI_mod.simulate_signal(acq_scheme, parameter_vector)
# add some noise
# E_real = E + np.random.normal(scale=0.01, size=(3, acq_scheme.number_of_measurements))
# E_imag = np.random.normal(scale=0.01, size=(3, acq_scheme.number_of_measurements))
# E = np.sqrt(E_real**2 + E_imag**2)
for i in range(0,3):
    plt.plot(acq_scheme.bvalues,E[i,:],'o')
#parameter_vector
#NODDI_mod.parameter_names



# simulate the image

simimg = np.zeros((dimx,dimy,np.shape(E)[1]))
nparam=6
param_map = np.zeros((dimx,dimy,nparam))
for x in range(0,dimx):
    for y in range(0,dimy):
        simimg[x,y,:] = E[int(simROIs[x,y]),:] + np.random.normal(0,0.02,nmeas)
        param_map[x,y,:]=parameter_vector[int(simROIs[x,y]),:]
plt.imshow(simimg[:,:,10])
fig, axs = plt.subplots(2, 4, figsize=[15, 10])
axs = axs.ravel()
for i in range(0,nparam):
    axs[i].imshow(param_map[:,:,i])



# fit bayes

nsteps = 2000
burn_in = 1000
#rearrange the data
data = np.reshape(simimg, (dimx*dimy, nmeas))
mask = np.reshape(simmask, dimx*dimy)
simROIs = np.reshape(simROIs,dimx*dimy)
simROIsmask = simROIs + 1
#priors defined over the whole image
#acceptance_rate, param_conv, params_all_new, params_all_orig, likelihood_stored, w_stored = fit_bayes.fit(NODDI_mod, acq_scheme, data, mask, nsteps, burn_in)
#priors defined over the ROIs
acceptance_rate_ROIs, param_conv_ROIs, params_all_new_ROIs, params_all_orig_ROIs, likelihood_stored_ROIs, w_stored_ROIs = fit_bayes.fit(NODDI_mod, acq_scheme, data, simROIsmask, nsteps, burn_in)
#reshape them