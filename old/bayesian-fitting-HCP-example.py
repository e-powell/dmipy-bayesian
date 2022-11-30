# load some necessary modules
from dmipy.core import modeling_framework
from os.path import join
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

#load the example data
from dmipy.data import saved_data
scheme_hcp, data_hcp = saved_data.wu_minn_hcp_coronal_slice()

#setup ball and stick model
from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.core.modeling_framework import MultiCompartmentModel
ball = gaussian_models.G1Ball()
stick = cylinder_models.C1Stick()
ballstick = MultiCompartmentModel(models=[stick, ball])

import asyncio
from importlib import reload

import fit_bayes
from fit_bayes import fit, tform_params  # , dict_to_array, array_to_dict
fit_bayes = reload(fit_bayes)

#setup stuff - want to move this inside the function

#get the mask
mask_hcp = (data_hcp[..., 0]>0)

#squash down into a 2D array - single slice
mask = np.squeeze(mask_hcp)
data = np.squeeze(data_hcp)

#reshape the data
nx = data.shape[0]
ny = data.shape[1]
ndw = data.shape[2]

from copy import copy, deepcopy
# generalise
model = deepcopy(ballstick)
data = np.reshape(data_hcp, (nx*ny, ndw))
mask = np.reshape(mask_hcp, nx*ny)


#get the mask
mask_hcp = (data_hcp[..., 0]>0)

#squash down into a 2D array - single slice
mask = np.squeeze(mask_hcp)
data = np.squeeze(data_hcp)

#reshape the data
nx = data.shape[0]
ny = data.shape[1]
ndw = data.shape[2]

from copy import copy, deepcopy
# generalise
model = deepcopy(ballstick)
data = np.reshape(data_hcp, (nx*ny, ndw))
mask = np.reshape(mask_hcp, nx*ny)

nsteps=500
burn_in=250

acceptance_rate, param_conv, params_all_new, params_all_orig, likelihood_stored, w_stored, lsq_fit = fit_bayes.fit(ballstick, scheme_hcp, data, mask, nsteps, burn_in)

params_img = {}
for i in params_all_new.keys():
    print(i)
    if ballstick.parameter_cardinality[i] == 1:
        params_img[i] = np.reshape(params_all_new[i],(nx,ny))
        params_img[i] = np.flip(params_img[i].transpose(1,0),0)
    else:
        print('not working yet! params_all_new seems too big for orientation parameters')

fig, axs = plt.subplots(2, 2, figsize=[10, 10])
axs = axs.ravel()

colormap=axs[0].imshow(params_img['partial_volume_0'],vmin=0, vmax=1)
fig.colorbar(colormap, ax=axs[0], shrink=0.8)
axs[0].set_title('partial_volume_0')
axs[0].set_axis_off()

colormap=axs[1].imshow(params_img['partial_volume_1'],vmin=0, vmax=1)
fig.colorbar(colormap, ax=axs[1], shrink=0.8)
axs[1].set_title('partial_volume_1')
axs[1].set_axis_off()

colormap=axs[2].imshow(params_img['C1Stick_1_lambda_par'],vmin=0, vmax=3e-9)
fig.colorbar(colormap, ax=axs[2], shrink=0.8)
axs[2].set_title('C1Stick_1_lambda_par')
axs[2].set_axis_off()

colormap=axs[3].imshow(params_img['G1Ball_1_lambda_iso'],vmin=0, vmax=3e-9)
fig.colorbar(colormap, ax=axs[3], shrink=0.8)
axs[3].set_title('G1Ball_1_lambda_iso')
axs[3].set_axis_off()

#lsq fit for comparison (could output this from the fit_bayes function?)
#BAS_fit_hcp = ballstick.fit(scheme_hcp, data_hcp, mask_hcp)
BAS_fit_hcp = lsq_fit

#plot lsq maps

'''
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#%matplotlib inline

fitted_parameters = BAS_fit_hcp.fitted_parameters

fig, axs = plt.subplots(2, 2, figsize=[10, 10])
axs = axs.ravel()

counter = 0
for name, values in fitted_parameters.items():
    if values.squeeze().ndim != 2:
        continue
    cf = axs[counter].imshow(values.squeeze().T, origin=True, interpolation='nearest')
    axs[counter].set_title(name)
    axs[counter].set_axis_off()
    fig.colorbar(cf, ax=axs[counter], shrink=0.8)
    counter += 1

#TO DO plot the inferred prior distributions
import scipy.stats as stats

fig, axs = plt.subplots(2, 3, figsize=[15, 10])
axs = axs.ravel()

names = ['D_par','D_iso','f_stick','inclination (0,$\pi$)','azimuth (-$\pi$,$\pi$)']


for i in range(0,5):
    marginal_mu = mu[i]
    variance = sigma[i,i]
    marginal_sigma = math.sqrt(variance)
    x = np.linspace(marginal_mu - 3*marginal_sigma, marginal_mu + 3*marginal_sigma, 100)
    axs[i].plot(x, stats.norm.pdf(x, marginal_mu, marginal_sigma))
    axs[i].set_title(names[i])
'''

from dmipy.signal_models import cylinder_models, gaussian_models
ball = gaussian_models.G1Ball()
stick = cylinder_models.C1Stick()
zeppelin = gaussian_models.G2Zeppelin()

from dmipy.distributions.distribute_models import SD1WatsonDistributed
watson_dispersed_bundle = SD1WatsonDistributed(models=[stick, zeppelin])

watson_dispersed_bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp','C1Stick_1_lambda_par','partial_volume_0')
watson_dispersed_bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
#see what happens when you don't fix the diffusivities!
#watson_dispersed_bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par', 1.7e-9)

from dmipy.core.modeling_framework import MultiCompartmentModel
NODDI_mod = MultiCompartmentModel(models=[ball, watson_dispersed_bundle])

#MAKE THE MASK A BIT SMALLER FOR TESTING
#mask_hcp[23:-1,:,:] = 0

model = deepcopy(NODDI_mod)

data = np.reshape(data_hcp, (nx*ny, ndw))
mask = np.reshape(mask_hcp, nx*ny)

nsteps=2000
burn_in=1000

# lsq_fit = model.fit(scheme_hcp, data, mask=mask)

import shelve
# load workspace variables
def load_workspace(filename):
    shelf = shelve.open(filename)
    print(shelf)
    for key in shelf:
        try:
            print(key)
            globals()[key] = shelf[key]
        except:
            print('ERROR shelving: {0}'.format(key))
    shelf.close()

load_workspace('noddi_lsq_fit.db')
acceptance_rate, param_conv, params_all_new, params_all_orig, likelihood_stored, w_stored, lsq_fit = fit_bayes.fit(NODDI_mod, scheme_hcp, data, lsq_fit, mask, nsteps, burn_in)

params_img = {}
params_img_orig = {}

for i in params_all_new.keys():
    print(i)
    if NODDI_mod.parameter_cardinality[i] == 1:
        params_img[i] = np.reshape(params_all_new[i],(nx,ny))
        params_img[i] = np.flip(params_img[i].transpose(1,0),0)

        params_img_orig[i] = np.reshape(params_all_orig[i],(nx,ny))
        params_img_orig[i] = np.flip(params_img_orig[i].transpose(1,0),0)


    else:
        print('not working yet! params_all_new seems too big for orientation parameters')
        #params_img[i] = np.reshape(params_all_new[i],(nx,ny,ballstick.parameter_cardinality[i]))

#BAYESIAN FIT

fig, axs = plt.subplots(3, 2, figsize=[10, 15])
axs = axs.ravel()

colormap=axs[0].imshow(params_img['SD1WatsonDistributed_1_SD1Watson_1_odi'],vmin=0, vmax=1)
fig.colorbar(colormap, ax=axs[0], shrink=0.8)
axs[0].set_title('SD1WatsonDistributed_1_SD1Watson_1_odi')
axs[0].set_axis_off()

colormap=axs[1].imshow(params_img['SD1WatsonDistributed_1_partial_volume_0'],vmin=0, vmax=1)
fig.colorbar(colormap, ax=axs[1], shrink=0.8)
axs[1].set_title('SD1WatsonDistributed_1_partial_volume_0')
axs[1].set_axis_off()

colormap=axs[2].imshow(params_img['partial_volume_1'],vmin=0, vmax=1)
fig.colorbar(colormap, ax=axs[2], shrink=0.8)
axs[2].set_title('partial_volume_1')
axs[2].set_axis_off()

colormap=axs[3].imshow(params_img['partial_volume_0'],vmin=0, vmax=1)
fig.colorbar(colormap, ax=axs[3], shrink=0.8)
axs[3].set_title('partial_volume_0')
axs[3].set_axis_off()

colormap=axs[4].imshow(params_img['G1Ball_1_lambda_iso'],vmin=0, vmax=3e-09)
fig.colorbar(colormap, ax=axs[4], shrink=0.8)
axs[4].set_title('G1Ball_1_lambda_iso')
axs[4].set_axis_off()

colormap=axs[5].imshow(params_img['SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par'],vmin=0, vmax=3e-09)
fig.colorbar(colormap, ax=axs[5], shrink=0.8)
axs[5].set_title('SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par')
axs[5].set_axis_off()